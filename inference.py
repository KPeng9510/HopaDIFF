import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset, Dataset_RHAS_random_clip
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter
# Define the seed value
seed = 123
import random
# Set seed for PyTorch
torch.manual_seed(seed)

# Set seed for CUDA (if using GPUs)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

# Set seed for Python's random module
random.seed(seed)

# Set seed for NumPy
np.random.seed(seed)

# Ensure deterministic behavior for PyTorch operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = 137 #len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)
        # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
        for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar
            print('val')
            val_result_dict = self.test(
                train_test_dataset, mode, device, label_dir,
                result_dir=result_dir, model_path=None, test=False)
            print('test')
            test_result_dict = self.test(
                test_test_dataset, mode, device, label_dir,
                result_dir=result_dir, model_path=None, test=True)
            for k,v in val_result_dict.items():
                print(f'Epoch {epoch} - {mode}-Val-{k} {v}')
            for k,v in test_result_dict.items():
                print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None):  
        
        #assert(test_dataset.mode == 'test')
        #assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        #assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])


        self.model.eval()
        self.model.to(device)


        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature, feature_orin, label, _ = test_dataset[video_idx]
            feature = torch.Tensor(feature).unsqueeze(0).permute(0,2,1)
            feature_orin = torch.Tensor(feature_orin).unsqueeze(0).permute(0,2,1)


            label = torch.Tensor(label)



            feature = feature.permute(0,2,1)
            output = [self.model.ddim_sample(feature.to(device),feature_orin.to(device), seed)] # output is a list of tuples
            output = [i.cpu() for i in output]
            left_offset = self.sample_rate // 2
            right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            output = output.mean(0).numpy()
            #print(output.shape)
            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[1]):
                    smoothed_output[:,c] = median_filter(output[:,c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            #output = np.argmax(output, 0)
            #print(output.shape)
            '''output = restore_full_sequence(output, 
                full_len=label.shape[-2], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )'''

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()

            #assert(output.shape == label.shape)
            
            return output, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, test=True):
        model_path = "/hkfs/work/workspace/scratch/fy2374-got/neurips2025/benchmarks/model_weights/holistic_partial_diffact_xlstmfusion_fourier_cond_rand_clip_true_2/50salads-Trained-S1/epoch-9-46.61655791658038.model"
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(device)    
        with torch.no_grad():
            all_pred = []
            all_label = []
            #
            for video_idx in tqdm(range(len(test_dataset))):
                '''if video_idx > 10:
                    break'''
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path)
                #print(pred.shape)
                #print(label.shape)
                pred = pred.transpose(1,0)
                all_pred.append(pred)
                all_label.append(label)
                #print(np.unique(np.argmax(pred,-1)))
                #if not os.path.exists(os.path.join(result_dir, 'prediction')):
                #    os.makedirs(os.path.join(result_dir, 'prediction'))

                '''file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                #file_ptr.write(' '.join(pred))
                file_ptr.close()'''
        video_list = [all_label,all_pred]

        import pickle as pkl
        if test == True:
            f = open("saved_ped_inference_clip/label_pred_test.pkl", "wb")
            pkl.dump(file=f, obj=video_list)
            f.close()
        else:
            f = open("saved_ped_inference_clip/label_pred_val.pkl", "wb")
            pkl.dump(file=f, obj=video_list)
            f.close()


        acc, edit, f1s = func_eval(
            label_dir, os.path.join(result_dir, 'prediction'), video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        
        return result_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    feature_dir = os.path.join(root_data_dir, dataset_name, 'features')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    event_list = np.arange(137)#np.loadtxt(mapping_file, dtype=str)
    #event_list = #[i[1] for i in event_list]
    num_classes = len(event_list)

    
    train_train_dataset = Dataset_RHAS_random_clip(split="train")#VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset =  Dataset_RHAS_random_clip(split="val")#VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = Dataset_RHAS_random_clip(split="test")#VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )   
    result_dir = "/hkfs/work/workspace/scratch/fy2374-got/neurips2025/benchmarks/model_weights/abl_no_bca2/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
        loss_weights, class_weighting, soft_label,
        num_epochs, batch_size, learning_rate, weight_decay,
        label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
        log_freq=log_freq, log_train_results=log_train_results
    )
