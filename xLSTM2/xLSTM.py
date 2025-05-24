import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D
from xLSTM.sLSTMblock import sLSTMblock


import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# bidirectional cross attention - have two sequences attend to each other with 1 attention step

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 2,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # relative positional bias, if supplied

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # mask

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # get attention along both sequence length and context length dimensions
        # shared similarity matrix

        attn = sim.softmax(dim = -1)
        context_attn = sim.softmax(dim = -2)

        # dropouts

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out







'''class sLSTMblock(nn.Module):
    def __init__(self, x_example, depth, dropout=0.2):
        super().__init__()
        self.input_size = x_example.shape[2]
        conv_channels = x_example.shape[1]
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.conv = CausalConv1D(self.input_size, self.input_size, int(self.input_size/8))
        self.drop = nn.Dropout(dropout)
        
        self.bca = BidirectionalCrossAttention(dim=self.input_size)

        self.i_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.f_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.o_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        self.z_gate = BlockDiagonal(self.input_size, self.input_size, depth)
        
        self.ri_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rf_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.ro_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rz_gate = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)

        self.ln_i = nn.LayerNorm(self.input_size)
        self.ln_f = nn.LayerNorm(self.input_size)
        self.ln_o = nn.LayerNorm(self.input_size)
        self.ln_z = nn.LayerNorm(self.input_size)
        
        self.GN = nn.LayerNorm(self.input_size)
        self.ln_c = nn.LayerNorm(self.input_size)
        self.ln_n = nn.LayerNorm(self.input_size)
        self.ln_h = nn.LayerNorm(self.input_size)
        
        self.left_linear = nn.Linear(self.input_size, int(self.input_size*(4/3)))
        self.right_linear = nn.Linear(self.input_size, int(self.input_size*(4/3)))

        self.ln_out = nn.LayerNorm(int(self.input_size*(4/3)))
        
        self.proj = nn.Linear(int(self.input_size*(4/3)), self.input_size)


        self.ln_2 = nn.LayerNorm(self.input_size)
        
        self.conv_2 = CausalConv1D(self.input_size, self.input_size, int(self.input_size/8))
        self.drop_2 = nn.Dropout(dropout)
        
        self.i_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth)
        self.f_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth)
        self.o_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth)
        self.z_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth)
        
        self.ri_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rf_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.ro_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)
        self.rz_gate_2 = BlockDiagonal(self.input_size, self.input_size, depth, bias=False)

        self.ln_i_2 = nn.LayerNorm(self.input_size)
        self.ln_f_2 = nn.LayerNorm(self.input_size)
        self.ln_o_2 = nn.LayerNorm(self.input_size)
        self.ln_z_2 = nn.LayerNorm(self.input_size)
        
        self.GN_2 = nn.LayerNorm(self.input_size)
        self.ln_c_2 = nn.LayerNorm(self.input_size)
        self.ln_n_2 = nn.LayerNorm(self.input_size)
        self.ln_h_2 = nn.LayerNorm(self.input_size)
        
        self.left_linear_2 = nn.Linear(self.input_size, int(self.input_size*(4/3)))
        self.right_linear_2 = nn.Linear(self.input_size, int(self.input_size*(4/3)))

        self.ln_out_2 = nn.LayerNorm(int(self.input_size*(4/3)))
        
        self.proj_2 = nn.Linear(int(self.input_size*(4/3)), self.input_size)

        self.init_states(x_example)
        
    def init_states(self, x):
        self.nt_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.ct_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.ht_1 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.mt_1 = torch.zeros(1, 1, x.shape[2], device=x.device)

        self.nt_2 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.ct_2 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.ht_2 = torch.zeros(1, 1, x.shape[2], device=x.device)
        self.mt_2 = torch.zeros(1, 1, x.shape[2], device=x.device)
    def forward(self, x_1, x_2):
        x_1 = self.ln(x_1)
        x_2 = self.ln_2(x_2)


        x_conv = F.silu(self.drop(self.conv( x_1.transpose(1, 2) ).transpose(1, 2)))
        x_conv_2 = F.silu(self.drop(self.conv_2(x_2.transpose(1, 2)).transpose(1, 2)))


        # start sLSTM
        ht_1 = self.ht_1
        ht_2 = self.ht_2


        i = torch.exp(self.ln_i( self.i_gate(x_conv) + self.ri_gate(ht_1) ) )
        f = torch.exp( self.ln_f(self.f_gate(x_conv) + self.rf_gate(ht_1) ) )

        i_2 = torch.exp(self.ln_i_2( self.i_gate_2(x_conv_2) + self.ri_gate_2(ht_2) ) )
        f_2 = torch.exp( self.ln_f_2(self.f_gate_2(x_conv_2) + self.rf_gate_2(ht_2) ) )


        f, f_2 = self.bca(f, f_2)





        m = torch.max(torch.log(f)+self.mt_1[:, 0, :].unsqueeze(1), torch.log(i))
        i = torch.exp(torch.log(i) - m)
        f = torch.exp(torch.log(f) + self.mt_1[:, 0, :].unsqueeze(1)-m)
        self.mt_1 = m.detach()
        
        m_2 = torch.max(torch.log(f_2)+self.mt_2[:, 0, :].unsqueeze(1), torch.log(i_2))
        i_2 = torch.exp(torch.log(i_2) - m_2)
        f_2 = torch.exp(torch.log(f_2) + self.mt_2[:, 0, :].unsqueeze(1)-m_2)
        self.mt_2 = m_2.detach()
        


        o = torch.sigmoid( self.ln_o(self.o_gate(x) + self.ro_gate(ht_1) ) )
        z = torch.tanh( self.ln_z(self.z_gate(x) + self.rz_gate(ht_1) ) )
        



        o_2 = torch.sigmoid( self.ln_o_2(self.o_gate_2(x_2) + self.ro_gate_2(ht_2) ) )
        z_2 = torch.tanh( self.ln_z_2(self.z_gate_2(x_2) + self.rz_gate_2(ht_2) ) )
        

        ct_1 = self.ct_1
        ct = f*ct_1 + i*z
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        
        ct_2 = self.ct_2
        ct_2 = f_2*ct_2 + i_2*z_2
        ct_2 = torch.mean(self.ln_c_2(ct_2), [0, 1], keepdim=True)
        self.ct_2 = ct_2.detach()
        

        nt_1 = self.nt_1
        nt = f*nt_1 + i
        nt = torch.mean(self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        

        nt_2 = self.nt_2
        nt_2 = f_2*nt_2 + i_2
        nt_2 = torch.mean(self.ln_n_2(nt_2), [0, 1], keepdim=True)
        self.nt_2 = nt_2.detach()


        ht = o*(ct/nt) # torch.Size([4, 8, 16])
        ht = torch.mean(self.ln_h(ht), [0, 1], keepdim=True)
        self.ht_1 = ht.detach()
        # end sLSTM


        ht_2 = o_2*(ct_2/nt_2) # torch.Size([4, 8, 16])
        ht_2 = torch.mean(self.ln_h_2(ht_2), [0, 1], keepdim=True)
        self.ht_2 = ht_2.detach()
        # end sLSTM


        slstm_out = self.GN(ht)
        
        left = self.left_linear(slstm_out)
        right = F.gelu(self.right_linear(slstm_out))
        
        out = self.ln_out(left*right)
        out = self.proj(out)


        slstm_out_2 = self.GN_2(ht_2)
        
        left_2 = self.left_linear_2(slstm_out_2)
        right_2 = F.gelu(self.right_linear_2(slstm_out_2))
        
        out_2 = self.ln_out_2(left_2*right_2)
        out_2 = self.proj_2(out_2)


        return out, out_2'''
  

class mLSTMblock(nn.Module):
    def __init__(self, x_example, factor, depth, dropout=0.2):
        super().__init__()
        self.input_size = x_example.shape[2]
        self.hidden_size = int(self.input_size*factor)
        self.bca = BidirectionalCrossAttention(dim=self.input_size*2)
        self.ln = nn.LayerNorm(self.input_size)
        
        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)
        
        self.conv = CausalConv1D(self.hidden_size, self.hidden_size, int(self.input_size/10)) 
        self.drop = nn.Dropout(dropout+0.1)
        
        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.dropq = nn.Dropout(dropout/2)
        self.dropk = nn.Dropout(dropout/2)
        self.dropv = nn.Dropout(dropout/2)
        
        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)
        
        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)

        self.lnl = nn.LayerNorm(self.hidden_size)


        
        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)

        self.drop2 = nn.Dropout(dropout)
        
        self.proj = nn.Linear(self.hidden_size, self.input_size)
        self.ln_proj = nn.LayerNorm(self.input_size)


        self.ln_2 = nn.LayerNorm(self.input_size)
        
        self.left_2 = nn.Linear(self.input_size, self.hidden_size)
        self.right_2 = nn.Linear(self.input_size, self.hidden_size)
        
        self.conv_2 = CausalConv1D(self.hidden_size, self.hidden_size, int(self.input_size/10)) 
        self.drop_2 = nn.Dropout(dropout+0.1)
        
        self.lskip_2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq_2 = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wk_2 = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wv_2 = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.dropq_2 = nn.Dropout(dropout/2)
        self.dropk_2 = nn.Dropout(dropout/2)
        self.dropv_2 = nn.Dropout(dropout/2)
        
        self.i_gate_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate_2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c_2 = nn.LayerNorm(self.hidden_size)
        self.ln_n_2 = nn.LayerNorm(self.hidden_size)
        
        self.lnf_2 = nn.LayerNorm(self.hidden_size)
        self.lno_2 = nn.LayerNorm(self.hidden_size)
        self.lni_2 = nn.LayerNorm(self.hidden_size)
        
        self.GN_2 = nn.LayerNorm(self.hidden_size)
        self.ln_out_2 = nn.LayerNorm(self.hidden_size)

        self.drop2_2 = nn.Dropout(dropout)
        
        self.proj_2 = nn.Linear(self.hidden_size, self.input_size)
        self.ln_proj_2 = nn.LayerNorm(self.input_size)
        



        
        self.init_states(x_example)
    
    def init_states(self, x_example):
        self.ct_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
        self.nt_1 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)

        self.ct_2 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
        self.nt_2 = torch.zeros([1, 1, self.hidden_size], device=x_example.device)
    
    def forward(self, x_1, x_2):
        assert x_1.ndim == 3
        
        x = self.ln(x_1) # layer norm on x
        x_2 = self.ln_2(x_2) # layer norm on x

        left = self.left(x) # part left 
        right = F.silu(self.right(x)) # part right with just swish (silu) function
        
        
        left_2 = self.left_2(x_2) # part left 
        right_2 = F.silu(self.right_2(x_2)) # part right with just swish (silu) function


        left_left = left.transpose(1, 2)
        left_left = F.silu( self.drop( self.conv( left_left ).transpose(1, 2) ) )
        l_skip = self.lskip(left_left)


        left_left_2 = left_2.transpose(1, 2)
        left_left_2 = F.silu( self.drop_2( self.conv_2(left_left_2).transpose(1, 2) ) )
        l_skip_2 = self.lskip_2(left_left_2)


        # start mLSTM
        q = self.dropq(self.wq(left_left))
        k = self.dropk(self.wk(left_left))
        v = self.dropv(self.wv(left))
        
        q_2 = self.dropq_2(self.wq_2(left_left_2))
        k_2 = self.dropk_2(self.wk_2(left_left_2))
        v_2 = self.dropv_2(self.wv_2(left_2))
        i_ = self.i_gate(left_left)
        i_2_ = self.i_gate_2(left_left_2)
        i_r,i_2_r = self.bca(i_,i_2_)
        i_2_ = 0.9*i_2_ + 0.1*i_2_r
        i_ = 0.9*i_ + 0.1*i_r
        i_2 = torch.exp(self.lni_2(i_2_))
        i = torch.exp(self.lni(i_))
        f = torch.exp(self.lnf(self.f_gate(left_left)))
        o = torch.sigmoid(self.lno(self.o_gate(left_left)))

        
        f_2 = torch.exp(self.lnf_2(self.f_gate_2(left_left_2)))
        o_2 = torch.sigmoid(self.lno_2(self.o_gate_2(left_left_2)))

        


        ct_1 = self.ct_1
        ct = f*ct_1 + i*v*k
        ct = torch.mean(self.ln_c(ct), [0, 1], keepdim=True)
        self.ct_1 = ct.detach()
        


        ct_2_2 = self.ct_2
        ct_2 = f_2*ct_2_2 + i_2*v_2*k_2
        ct_2 = torch.mean(self.ln_c_2(ct_2), [0, 1], keepdim=True)
        self.ct_2 = ct_2.detach()
        


        nt_1 = self.nt_1
        nt = f*nt_1 + i*k
        nt =torch.mean( self.ln_n(nt), [0, 1], keepdim=True)
        self.nt_1 = nt.detach()
        




        nt_2_2 = self.nt_2
        nt_2 = f*nt_2_2 + i_2*k_2
        nt_2 =torch.mean(self.ln_n_2(nt_2), [0, 1], keepdim=True)
        self.nt_2 = nt_2.detach()
        


        ht = o * ((ct*q) / torch.max(nt*q))
        # end mLSTM
        ht = ht
        

        ht_2 = o_2 * ((ct_2*q_2) / torch.max(nt_2*q_2))
        # end mLSTM
        ht_2 = ht_2


        left = self.drop2(self.GN(ht + l_skip))
        
        out = self.ln_out(left * right)
        out = self.ln_proj(self.proj(out))



        left_2 = self.drop2_2(self.GN_2(ht_2 + l_skip_2))
        
        out_2 = self.ln_out_2(left_2 * right_2)
        out_2 = self.ln_proj_2(self.proj_2(out_2))       
        return out, out_2

class xLSTM(nn.Module):
    def __init__(self, layers, x_example, depth=4, factor=2):
        super(xLSTM, self).__init__()

        self.layers = nn.ModuleList()
        self.layers_name = layers
        for layer_type in layers:
            if layer_type == 's':
                self.layer = nn.ModuleList([sLSTMblock(x_example, depth).cuda(),sLSTMblock(x_example, depth).cuda()])
            elif layer_type == 'm':
                layer = mLSTMblock(x_example, factor, depth)
                self.layers.append(layer)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            
    
    def init_states(self, x):
        [l.init_states(x) for l in self.layers]
        
    def forward(self, x_1, x_2):
        x_original_1 = x_1.clone()
        x_original_2 = x_2.clone()
        for layer in self.layers_name:
            if layer == 'm':
                x_1, x_2 = self.layers[0](x_1, x_2)
               
                x_1 += x_original_1
                x_2 += x_original_2
            else:
                x_1 = self.layer[0](x_1) + x_original_1
                x_2 = self.layer[0](x_2) + x_original_2
        return x_1, x_2
