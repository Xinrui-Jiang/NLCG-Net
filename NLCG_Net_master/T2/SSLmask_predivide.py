import os
import time
import numpy as np
import scipy.io as scio
import torch
import parser_ops
import fastmri
from torch.utils.data import DataLoader
import utils, UnrollNet
from modules import RMSELoss, Dataset, Dataset_Inference, train, validation, test
import matplotlib.pyplot as plt
import data_consistency as dc

parser = parser_ops.get_parser()
args = parser.parse_args([])

torch.manual_seed(3407)
np.random.seed(3407)
pretrain_iter = 5

#0. Load the data
acc_mask = torch.load(os.path.join(args.data_dir,"confirmed_t2_ACC{},noACS.pth".format(args.acc_rate)))    #(256,208,12,8)
kspace_data = torch.load(os.path.join(args.data_dir,args.kdata_name)).to(torch.complex64)#(256,208,8,12)
kspace_data = kspace_data * acc_mask.permute(0,1,3,2)
kspace_data = kspace_data / args.k_scaling_factor

sensitivity_maps = torch.from_numpy(scio.loadmat(os.path.join(args.data_dir,args.sens_name))['sens']).to(torch.complex64)#(256,208,12)

print(kspace_data.shape,acc_mask.shape)
H,W,N_time,n_coil = kspace_data.shape
print('H: {}, W: {}, N_time: {}, n_coil: {}'.format(H,W,N_time,n_coil))
print('current echo time: {}, r_scaling_factor: {}'.format(args.echo_time_list,args.r_scaling_factor))

#2. Create train and validation mask. 4 kinds of mask, used in train(all_trn,loss_mask) and validation (trn,val)
##### ..................Generate validation and training mask....................................
#8:2correspond to trn_mask(remainer_mask), val_mask
#0.8,     0.2
trn_mask, val_mask = utils.uniform_selection(kspace_data,acc_mask, rho=args.rho_val)
remainer_mask = trn_mask.detach().clone()
trn_mask, val_mask = trn_mask.unsqueeze(0), val_mask.unsqueeze(0)

all_trn_mask, loss_mask = torch.empty((args.num_reps, H, W,n_coil,N_time)), torch.empty((args.num_reps, H, W,n_coil,N_time))
for jj in range(args.num_reps):
    all_trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_data,remainer_mask,rho=args.rho_train)  # mask:(25,640,368)

#..............................put all mask to device..................................
trn_mask,val_mask,all_trn_mask,loss_mask,acc_mask = trn_mask.to(args.device),val_mask.to(args.device),all_trn_mask.to(args.device),loss_mask.to(args.device),acc_mask.to(args.device)
kspace_data,sensitivity_maps = kspace_data.to(args.device),sensitivity_maps.to(args.device)

#3. Mask all data
#..............................creating initial M,T2 data..................................
val_input,trn_input = torch.ones((1,H,W,3),dtype=torch.float32).to(args.device),torch.ones((args.num_reps, H, W, 3), dtype=torch.float32).to(args.device)
val_input[...,2],trn_input[...,2] = val_input[...,2]/args.r_scaling_factor,trn_input[...,2]/args.r_scaling_factor
val_input[...,:2],trn_input[...,:2] = val_input[...,:2]*args.k_scaling_factor,trn_input[...,:2]*args.k_scaling_factor

#..............................creating initial kspace data..................................
val_ref_kspace = torch.empty(kspace_data.shape,dtype=torch.complex64).to(args.device)  #(256,208,31,12)
val_input_k = torch.empty(kspace_data.shape,dtype=torch.complex64).to(args.device)     #(256,208,31,12)

trn_ref_kspace = torch.empty((args.num_reps, H, W, N_time, n_coil),dtype=torch.complex64).to(args.device)  # ref_kspace:(25,256,208,31,12),Loss
trn_input_k = torch.empty((args.num_reps, H, W, N_time, n_coil), dtype=torch.complex64).to(args.device)  # (25,256,208,31,12)

for i in range(N_time):
    val_input_k[...,i,:] = kspace_data[...,i,:] * trn_mask[0,...,i]
    val_ref_kspace[...,i,:] = kspace_data[...,i,:] * val_mask[0,...,i]

for jj in range(args.num_reps):
    for i in range(N_time):
        trn_input_k[jj, ..., i, :] = kspace_data[..., i, :] * all_trn_mask[jj,...,i]
        trn_ref_kspace[jj, ..., i, :] = kspace_data[..., i, :] * loss_mask[jj,...,i]

#3. Pretrain data
# %% Prepare the data for the training
sensitivity_maps = torch.tile(sensitivity_maps.unsqueeze(0),(args.num_reps,1,1,1))                 #(25,256,208,12)
trn_ref_kspace = torch.stack([trn_ref_kspace.real,trn_ref_kspace.imag],dim=-1)          #ref_kspace:(25,256,208,31,12,2)
val_ref_kspace = torch.stack([val_ref_kspace.real,val_ref_kspace.imag],dim=-1).unsqueeze(0)  #(1,256,208,31,12,2)
val_input_k = val_input_k.unsqueeze(0)            # (1,25,256,208,31,12)

#Pretrain to get a raw data
trn_out = torch.empty(trn_input.shape,dtype=torch.float32)
val_out = torch.empty(val_input.shape,dtype=torch.float32)

for i in range(args.num_reps):
      trn_out[i] = dc.NLCG(kimage_Data=trn_input_k[i],sens_map=sensitivity_maps[i],mask=all_trn_mask[i],M=trn_input[i],z=None,iterations=pretrain_iter,lam=0.0)

val_out = dc.NLCG(kimage_Data=val_input_k[0],sens_map=sensitivity_maps[0],mask=trn_mask[0],M=val_input[0],z=None,iterations=pretrain_iter,lam=0.0).unsqueeze(0)

directory = args.SSLdata_dir

if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(trn_out.cpu(),os.path.join(directory, 'trn_out.pth'))
torch.save(trn_input_k.cpu(),os.path.join(directory, 'trn_input_k.pth'))
torch.save(all_trn_mask.cpu(),os.path.join(directory, 'all_trn_mask.pth'))
torch.save(loss_mask.cpu(),os.path.join(directory, 'loss_mask.pth'))
torch.save(trn_ref_kspace.cpu(),os.path.join(directory, 'trn_ref_kspace.pth'))

torch.save(val_out.cpu(),os.path.join(directory, 'val_out.pth'))
torch.save(val_input_k.cpu(),os.path.join(directory, 'val_input_k.pth'))
torch.save(trn_mask.cpu(),os.path.join(directory, 'trn_mask.pth'))
torch.save(val_mask.cpu(),os.path.join(directory, 'val_mask.pth'))
torch.save(val_ref_kspace.cpu(),os.path.join(directory, 'val_ref_kspace.pth'))