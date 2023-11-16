import torch.nn as nn
import torch
#from ResNet import ResNet
from full_UNet import U_net
import data_consistency as dc
import time
from utils import kspace_transform as k_trans

max_val_start = torch.ones((1,3),dtype=torch.float32)
max_val_start[:,2] = max_val_start[:,2] * 33.0793
#max_val_start[:,:2] = max_val_start[:,:2] * 0.2
lamda_start = 0.05

class UnrolledNet(nn.Module):

    def __init__(self,batchSize,CG_iter,nb_unroll_blocks=4,device=torch.device('cuda')):
        super(UnrolledNet, self).__init__()
        #self.scaling_val = scaling_val
        self.batchSize = batchSize
        self.CG_iter = CG_iter
        self.lam = nn.Parameter(torch.tensor([lamda_start]))
        self.nb_unroll_blocks = nb_unroll_blocks
        #self.regularizer = ResNet(device, in_ch=3, num_of_resblocks=nb_res_blocks)
        self.regularizer = U_net(in_ch=3)
        #self.Rregularizer = U_net(in_ch=2)
        
        self.max_val = nn.Parameter(max_val_start)
        
        print(self.lam.is_leaf,self.lam.requires_grad)
        print(self.max_val.is_leaf,self.max_val.requires_grad)


    def forward(self, kimages,input_x, trn_mask, loss_mask, sens_maps):
        x = input_x                                                         #shape of x: (1,256,208,3)
        
        for _ in range(self.nb_unroll_blocks):
            regu_x = self.regularizer((x/self.max_val).permute(0,3,1,2)).permute(0,2,3,1)
            
            x = dc.dc_block(kimages=kimages,sens_maps=sens_maps,mask=trn_mask,M=x,lam=self.lam,z=regu_x,iterations=self.CG_iter,batchSize=self.batchSize)
            
        #print('M max: {}, M min: {}'.format(torch.max(x[...,:2]), torch.min(x[...,:2])))
        #print('R max: {}, R min: {}, R mean: {}'.format(torch.max(x[...,2]), torch.min(x[...,2]),torch.mean(x[...,2])))
        nw_kspace_output = k_trans(x,sens_maps,loss_mask,N_time=8)

        return x, self.lam, nw_kspace_output