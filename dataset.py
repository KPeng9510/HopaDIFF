import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
#from utils import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d
import pickle as pkl

def load_feature(feature_dir, video, transpose):
    file_name = os.path.join(feature_dir, video+'.npy')
    feature = np.load(file_name)

    if transpose:
        feature = feature.T
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    
    return feature #[::sample_rate]

def load_action_mapping(map_fname, sep=" "):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            tokens = line.split(sep)
            l = sep.join(tokens[1:])
            i = int(tokens[0])
            label2index[l] = i
            index2label[i] = l

    return label2index, index2label
def list_filenames_in_folder(folder_path):
    """
    Returns a list of filenames in the specified folder (non-recursive).

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list: Filenames (not including full path).
    """
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
class ListBatchDataLoader:
    def __init__(self, dataset, batch_size=4, shuffle=True):
        """
        Custom dataloader that returns each batch as a list of samples.

        Args:
            dataset (list or Dataset): Your dataset.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.current_idx:end_idx]
        feat, labels, segs =  [], [], []
        for i in batch_indices:
            feat.append(torch.Tensor(self.dataset[i][0]))
            #print(self.dataset[i][1])
            labels.append(self.dataset[i][1])
            segs.append(self.dataset[i][2])
        #batch = [self.dataset[i] for i in batch_indices]
        self.current_idx = end_idx
        return feat, labels,segs

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size



def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    frame_wise_labels = frame_wise_labels.numpy()
    last_label = frame_wise_labels[0]
    labels.append(frame_wise_labels[0])
    starts.append(0)
    for i in range(len(frame_wise_labels)):
        #print(frame_wise_labels[i])
        #print(frame_wise_labels[i]!= last_label)
        if np.sum(frame_wise_labels[i] != last_label):
            labels.append(frame_wise_labels[i])
            starts.append(i)
            ends.append(i)
            last_label = frame_wise_labels[i]
    ends.append(i + 1)
    return labels, starts, ends




def get_data_dict(feature_dir, label_dir, video_list, event_list, sample_rate=4, temporal_aug=True, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = os.path.join(feature_dir, '{}.npy'.format(video))
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        feature = np.load(feature_file, allow_pickle=True)
        
        if len(feature.shape) == 3:
            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
        else:
            raise Exception('Invalid Feature.')
                    
        assert(feature.shape[1] == event_seq_raw.shape[0])
        assert(feature.shape[1] == boundary_seq_raw.shape[0])
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        else:
            feature = [feature[:,::sample_rate,:]]  
            event_seq_ext = [event_seq_raw[::sample_rate]]
            boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

        data_dict[video]['feature'] = [torch.from_numpy(i).float() for i in feature]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['event_seq_ext'] = [torch.from_numpy(i).float() for i in event_seq_ext]
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()
        data_dict[video]['boundary_seq_ext'] = [torch.from_numpy(i).float() for i in boundary_seq_ext]
        
    return data_dict

def get_boundary_seq(event_seq, boundary_smooth=None):

    boundary_seq = np.zeros(len(event_seq))
    #print(event_seq)
    _, start_times, end_times = get_labels_start_end_time(event_seq)
    #print(start_times)
    #print(boundary_seq.shape)
    boundaries = start_times[1:]
    #assert min(boundaries) > 0
    boundary_seq[boundaries] = 1
    boundary_seq[[i-1 for i in boundaries]] = 1

    if boundary_smooth is not None:
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
        
        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        temp_seq[temp_seq.shape[0] // 2] = 1
        temp_seq[temp_seq.shape[0] // 2 - 1] = 1
        norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        boundary_seq /= boundary_seq.max()

    return boundary_seq


def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
        
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)
    print(frame_ticks.shape)
    print(x.shape)
    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out


class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, class_num, mode):
        super(VideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]
        
    def get_class_weights(self):
        
        full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)

        return class_weights

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_ext']
            boundary = self.data_dict[video]['boundary_seq_ext']

            temporal_aug_num = len(feature)
            temporal_rid = random.randint(0, temporal_aug_num - 1) # a<=x<=b
            feature = feature[temporal_rid]
            label = label[temporal_rid]
            boundary = boundary[temporal_rid]

            spatial_aug_num = feature.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1) # a<=x<=b
            feature = feature[spatial_rid]
            
            feature = feature.T   # F x T

            boundary = boundary.unsqueeze(0)
            boundary /= boundary.max()  # normalize again
            
        if self.mode == 'test':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_raw']
            boundary = self.data_dict[video]['boundary_seq_ext']  # boundary_seq_raw not used

            feature = [torch.swapaxes(i, 1, 2) for i in feature]  # [10 x F x T]
            label = label.unsqueeze(0)   # 1 X T'  
            boundary = [i.unsqueeze(0).unsqueeze(0) for i in boundary]   # [1 x 1 x T]  

        return feature, label, boundary, video

    
class Dataset_RHAS(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, nclasses=137, split="train"):
        """
        """
        self.split = split
        if self.split == "train":
            self.feat_path = "/hkfs/work/workspace/scratch/fy2374-got/neurips2025/RHAS_feature_blipv2/"
        elif self.split == "val":
            self.feat_path = "/hkfs/work/workspace/scratch/fy2374-got/neurips2025/RHAS_feature_blipv2_val/"
        else:
            self.feat_path = "/hkfs/work/workspace/scratch/fy2374-got/neurips2025/RHAS_feature_blipv2_test/"
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/actions_"+self.split+"_rhas.pkl", "rb")
        self.actions = pkl.load(f)
        #print(len(self.actions))
        f.close()
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/frame_ids_"+self.split+"_rhas.pkl", "rb")
        self.frame_ids = pkl.load(f)
        f.close()
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/person_ref_"+self.split+"_rhas.pkl", "rb")
        self.person_ref = pkl.load(f)
        f.close()
        # store dataset information
        self.nclasses = 137
        ######conduct file checking
        #print(self.actions[0])
        files_names = list_filenames_in_folder(self.feat_path)
        index_list = []
        for name in files_names:
            index_list.append(int(name.split("/")[-1].split("_")[0]))


        #self.actions = [self.actions[ind] for ind in index_list]
        #self.frame_ids = [self.frame_ids[ind] for ind in index_list]
        #self.person_ref = [self.person_ref[ind] for ind in index_list]
        self.index_list = index_list

    def one_hot_encode(self, labels, num_classes=None):
        """
        Generate one-hot encoded labels for a multi-class classification problem.

        Args:
            labels (list or array): List or array of integer class labels (e.g., [0, 2, 1, 3]).
            num_classes (int, optional): Total number of classes. If None, inferred from labels.

        Returns:
            np.ndarray: One-hot encoded matrix of shape (len(labels), num_classes).
        """
        labels = np.array(labels)
        if num_classes is None:
            num_classes = np.max(labels) + 1

        one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot        

    
    def __repr__(self):
        return str(self)

    def get_all_labels(self,):

        all_labels = []
        for l_act in self.actions:
            annotations = [torch.Tensor(self.one_hot_encode(label_action,self.nclasses)) for label_action in l_act]
            all_labels.extend(annotations)
        return all_labels

    def __getitem__(self, ind):
        index = self.index_list[ind]
        label_actions = self.actions[index]
        #print(label_actions)
        feat_name = self.feat_path +  str(index) + "_blip_feat.npy"
        original_path = feat_name
        if "/RHAS_feature_blipv2/" in feat_name:
            feat_name = feat_name.replace("/RHAS_feature_blipv2/", "/RHAS_feature_blipv2_train_grounding_dino/")
        elif "/RHAS_feature_blipv2_val/" in feat_name:
            feat_name = feat_name.replace("/RHAS_feature_blipv2_val/", "/RHAS_feature_blipv2_val_grounding_dino/")
        elif "/RHAS_feature_blipv2_test/" in feat_name:
            feat_name = feat_name.replace("/RHAS_feature_blipv2_test/", "/RHAS_feature_blipv2_test_grounding_dino/")
        #if self.split == "train":
        #    feat_name = feat_name.replace("RHAS_feature_blipv2","RHAS_feature_blipv2_train_grounding_dino")
        #elif self.split == "val":
        #    feat_name = feat_name.replace("RHAS_feature_blipv2_val","RHAS_feature_blipv2_val_grounding_dino")
        #else:
        #    feat_name = feat_name.replace("RHAS_feature_blipv2_test","RHAS_feature_blipv2_test_grounding_dino")
        try:
            video_features = np.load(feat_name)
        except:
            video_features = np.load(original_path)

        video_features_original = np.load(original_path)
        #print(video_features_original.shape)
        #print(video_features.shape)
        #start_idx = np.random.randint(low=0, high=max(len(video_features)-self.random_fenster, 1), size=1)[0]
        #end_idx = start_idx + self.random_fenster
        #print(label_actions)
        # print(len(label_actions))
        annotations = [torch.Tensor(self.one_hot_encode(label_action,self.nclasses)) for label_action in label_actions]
        #print(len(annotations))
        #try:
        all_anno = []
        for anno in annotations:
            if anno.shape[0] >= 2:
                anno = anno.sum(0)
                all_anno.append(anno.squeeze())
            else:
                all_anno.append(anno.squeeze())
        annotations = torch.stack(all_anno)
        vid_feat = video_features.mean(1)
        vid_feat_original = video_features_original.mean(1)
        boundaries = get_boundary_seq(annotations[:len(video_features)])
        #print(boundaries.shape)
        #noise = np.random.normal(0.0, 0.01, vid_feat.shape)
        if self.split != 'train':
            return vid_feat, vid_feat_original[:len(video_features)], annotations[:len(video_features)], boundaries
        else:
            return vid_feat , vid_feat_original[:len(video_features)], annotations[:len(video_features)], boundaries



    def __len__(self):
        return len(self.index_list)



class Dataset_RHAS_random(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, nclasses=137, split="train"):
        """
        """
        self.split = split
        phase = self.split
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/randomsplits/"+phase+"_files.pkl", "rb")
        self.files = pkl.load(f)
        f.close()
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/randomsplits/"+phase+"_anno.pkl", "rb")
        self.actions = pkl.load(f)
        f.close()
        # store dataset information
        self.nclasses = 137
        ######conduct file checking


    def one_hot_encode(self, labels, num_classes=None):
        """
        Generate one-hot encoded labels for a multi-class classification problem.

        Args:
            labels (list or array): List or array of integer class labels (e.g., [0, 2, 1, 3]).
            num_classes (int, optional): Total number of classes. If None, inferred from labels.

        Returns:
            np.ndarray: One-hot encoded matrix of shape (len(labels), num_classes).
        """
        labels = np.array(labels)
        if num_classes is None:
            num_classes = np.max(labels) + 1

        one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot        

    
    def __repr__(self):
        return str(self)

    def get_all_labels(self,):

        all_labels = []
        for l_act in self.actions:
            annotations = [torch.Tensor(self.one_hot_encode(label_action,self.nclasses)) for label_action in l_act]
            all_labels.extend(annotations)
        return all_labels

    def __getitem__(self, ind):
        #index = self.index_list[ind]
        label_actions = self.actions[ind]
        feat_name = self.files[ind] #self.feat_path +  str(index) + "_blip_feat.npy"
        original_path = feat_name
        if "/RHAS_feature_blipv2/" in feat_name:
            feat_name = feat_name.replace("/RHAS_feature_blipv2/", "/RHAS_feature_blipv2_train_grounding_dino/")
        elif "/RHAS_feature_blipv2_val/" in feat_name:
            feat_name = feat_name.replace("/RHAS_feature_blipv2_val/", "/RHAS_feature_blipv2_val_grounding_dino/")
        elif "/RHAS_feature_blipv2_test/" in feat_name:
            feat_name = feat_name.replace("/RHAS_feature_blipv2_test/", "/RHAS_feature_blipv2_test_grounding_dino/")
        #if self.split == "train":
        #    feat_name = feat_name.replace("RHAS_feature_blipv2","RHAS_feature_blipv2_train_grounding_dino")
        #elif self.split == "val":
        #    feat_name = feat_name.replace("RHAS_feature_blipv2_val","RHAS_feature_blipv2_val_grounding_dino")
        #else:
        #    feat_name = feat_name.replace("RHAS_feature_blipv2_test","RHAS_feature_blipv2_test_grounding_dino")
        try:
            video_features = np.load(feat_name)
        except:
            video_features = np.load(original_path)

        video_features_original = np.load(original_path)
        #print(video_features_original.shape)
        #print(video_features.shape)
        #start_idx = np.random.randint(low=0, high=max(len(video_features)-self.random_fenster, 1), size=1)[0]
        #end_idx = start_idx + self.random_fenster
        #print(label_actions)
        # print(len(label_actions))
        annotations = [torch.Tensor(self.one_hot_encode(label_action,self.nclasses)) for label_action in label_actions]
        #print(len(annotations))
        #try:
        all_anno = []
        for anno in annotations:
            if anno.shape[0] >= 2:
                anno = anno.sum(0)
                all_anno.append(anno.squeeze())
            else:
                all_anno.append(anno.squeeze())
        annotations = torch.stack(all_anno)
        vid_feat = video_features.mean(1)
        vid_feat_original = video_features_original.mean(1)
        boundaries = get_boundary_seq(annotations[:len(video_features)])
        #print(boundaries.shape)
        #noise = np.random.normal(0.0, 0.01, vid_feat.shape)
        if self.split != 'train':
            return vid_feat, vid_feat_original[:len(video_features)], annotations[:len(video_features)], boundaries
        else:
            return vid_feat , vid_feat_original[:len(video_features)], annotations[:len(video_features)], boundaries


        '''if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]'''

    def __len__(self):
        return len(self.actions)
class Dataset_RHAS_random_clip(object):
    """
    self.features[video]: the feature array of the given video (frames x dimension)
    self.input_dimension: dimension of video features
    self.n_classes: number of classes
    """

    def __init__(self, nclasses=137, split="train"):
        """
        """
        self.split = split
        phase = self.split
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/randomsplits/"+phase+"_files.pkl", "rb")
        self.files = pkl.load(f)
        f.close()
        f = open("/hkfs/work/workspace/scratch/fy2374-got/neurips2025/Processing_codes/randomsplits/"+phase+"_anno.pkl", "rb")
        self.actions = pkl.load(f)
        f.close()
        # store dataset information
        self.nclasses = 137
        ######conduct file checking


    def one_hot_encode(self, labels, num_classes=None):
        """
        Generate one-hot encoded labels for a multi-class classification problem.

        Args:
            labels (list or array): List or array of integer class labels (e.g., [0, 2, 1, 3]).
            num_classes (int, optional): Total number of classes. If None, inferred from labels.

        Returns:
            np.ndarray: One-hot encoded matrix of shape (len(labels), num_classes).
        """
        labels = np.array(labels)
        if num_classes is None:
            num_classes = np.max(labels) + 1

        one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
        one_hot[np.arange(labels.size), labels] = 1.0
        return one_hot        

    
    def __repr__(self):
        return str(self)

    def get_all_labels(self,):

        all_labels = []
        for l_act in self.actions:
            annotations = [torch.Tensor(self.one_hot_encode(label_action,self.nclasses)) for label_action in l_act]
            all_labels.extend(annotations)
        return all_labels

    def __getitem__(self, ind):
        #index = self.index_list[ind]
        label_actions = self.actions[ind]
        feat_name = self.files[ind] #self.feat_path +  str(index) + "_blip_feat.npy"
        original_path = feat_name
        if "/RHAS_feature_blipv2/" in original_path:
            original_path = original_path.replace("/RHAS_feature_blipv2/", "/RHAS_1_feature_clip_train/RHAS_feature_clip_train").replace('blip', 'clip')
            feat_name = feat_name.replace("/RHAS_feature_blipv2/","/RHAS_feature_clip_train_grounding_dino/").replace('blip', 'clip')

        elif "/RHAS_feature_blipv2_val/" in original_path:
            original_path = original_path.replace("/RHAS_feature_blipv2_val/", "/RHAS_1_feature_clip_val/RHAS_feature_clip_val").replace('blip', 'clip')
            feat_name = feat_name.replace("/RHAS_feature_blipv2_val/","/RHAS_feature_clip_val_grounding_dino/").replace('blip', 'clip')

        elif "/RHAS_feature_blipv2_test/" in original_path:
            original_path = original_path.replace("/RHAS_feature_blipv2_test/", "/RHAS_1_feature_clip_test/RHAS_feature_clip_test").replace('blip', 'clip')
            feat_name = feat_name.replace("/RHAS_feature_blipv2_test/","/RHAS_feature_clip_test_grounding_dino/").replace('blip', 'clip')

        try:
            video_features = np.load(feat_name)
        except:
            video_features = np.load(original_path)
        #start_idx = np.random.randint(low=0, high=max(len(video_features)-self.random_fenster, 1), size=1)[0]
        #end_idx = start_idx + self.random_fenster
        #print(label_actions)
        # print(len(label_actions))
        annotations = [torch.Tensor(self.one_hot_encode(label_action,self.nclasses)) for label_action in label_actions]
        #print(len(annotations))
        #try:
        video_features_original = np.load(original_path)
        #print(video_features_original.shape)
        all_anno = []
        for anno in annotations:
            if anno.shape[0] >= 2:
                anno = anno.sum(0)
                all_anno.append(anno.squeeze())
            else:
                all_anno.append(anno.squeeze())
        annotations = torch.stack(all_anno)
        #vid_feat = video_features.mean(1)
        #vid_feat_original = video_features_original.mean(1)
        boundaries = get_boundary_seq(annotations[:len(video_features)])
        #print(boundaries.shape)
        #noise = np.random.normal(0.0, 0.01, vid_feat.shape)
        if self.split != 'train':
            return torch.Tensor(video_features).permute(0,1),torch.Tensor(video_features_original[:len(video_features)]).permute(1,0), annotations[:len(video_features)], boundaries
        
        else:
           
            return torch.Tensor(video_features).permute(1,0) ,torch.Tensor(video_features_original[:len(video_features)]).permute(1,0), annotations[:len(video_features)], boundaries


        '''if video not in self.video_list:
            raise ValueError(video)

        if video not in self.data:
            self.data[video] = self.load_video(video)

        return self.data[video]'''

    def __len__(self):
        return len(self.actions)