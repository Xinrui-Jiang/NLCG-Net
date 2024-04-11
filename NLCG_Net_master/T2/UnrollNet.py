import torch.nn as nn
import torch
from ResNet import ResNet
import parser_ops
from UNet import U_net
import data_consistency as dc
import time
from utils import kspace_transform as k_trans
parser = parser_ops.get_parser()
args = parser.parse_args([])

lamda_start = 0.05

class UnrolledNet(nn.Module):

    def __init__(self,batchSize,CG_iter,nb_unroll_blocks=4,device=torch.device('cuda')):
        super(UnrolledNet, self).__init__()
        self.batchSize = batchSize
        self.CG_iter = CG_iter
        self.lam = nn.Parameter(torch.tensor([lamda_start]))
        self.nb_unroll_blocks = nb_unroll_blocks
        if args.model_type=='ResNet':
          self.regularizer = ResNet(device, in_ch=3, num_of_resblocks=args.nb_res_blocks)
        elif args.model_type=='UNet':
          self.regularizer = U_net(in_ch=3)
        else:
          raise Exception("Not a Valid Specified Model Type")
        print(self.lam.is_leaf,self.lam.requires_grad)

    def forward(self, kimages,input_x, trn_mask, loss_mask, sens_maps):
        x = input_x                                                         #shape of x: (1,256,208,3)
        for _ in range(self.nb_unroll_blocks):
            regu_x = self.regularizer(x.permute(0,3,1,2)).permute(0,2,3,1)
            x = dc.dc_block(kimages=kimages,sens_maps=sens_maps,mask=trn_mask,M=x,lam=self.lam,z=regu_x,iterations=self.CG_iter,batchSize=self.batchSize)
            
        print('M max: {}, M min: {}, M mean: {}'.format(torch.max(x[...,:2]), torch.min(x[...,:2]),torch.mean(x[...,:2])))
        print('R max: {}, R min: {}, R mean: {}'.format(torch.max(x[...,2]), torch.min(x[...,2]),torch.mean(x[...,2])))
        nw_kspace_output = k_trans(x,sens_maps,loss_mask,N_time=args.necho_GLOB)

        return x, self.lam, nw_kspace_output