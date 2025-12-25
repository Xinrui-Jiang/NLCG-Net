import os
import time
import numpy as np
import scipy.io as scio
import torch
import fastmri
from torch.utils.data import DataLoader
import utils, UnrollNet
from modules import Dataset, Dataset_Inference, train, validation, test
import data_consistency as dc

#1. Load the data
data_dirt = r"/root/autodl-tmp/MOBA-T2mapping-master/T1 mapping dataset/"

sensitivity_maps = torch.load(os.path.join(data_dirt,"sens_map_11.pth"))                #192,158,32
acc_mask = torch.load(os.path.join(data_dirt,"ACC2,noACS.pth"))          #192,158,32,5
kspace_data = torch.load(os.path.join(data_dirt,"5TIimg_11.pth"))        #192,158,32,5
kspace_data = kspace_data * acc_mask

print('max value: {}'.format(torch.max(torch.abs(kspace_data))))
kspace_data = kspace_data / torch.max(torch.abs(kspace_data))
kspace_data = kspace_data.permute(0,1,3,2) 

print(acc_mask.shape)
print(kspace_data.shape)                 #192,158,5,32
time.sleep(5)

H,W,N_time,n_coil = kspace_data.shape
rho_val,rho_train = 0.2, 0.4
num_reps = 25
batchSize = 1
acc_rate = 2
device = 'cuda'

pretrain_iter = 800

torch.manual_seed(3407)
np.random.seed(3407)

#2. Create train and validation mask. 4 kinds of mask, used in train(all_trn,loss_mask) and validation (trn,val)
##### ..................Generate validation and training mask....................................
#8:2correspond to trn_mask(remainer_mask), val_mask
#0.8,     0.2
trn_mask, val_mask = utils.uniform_selection(kspace_data,acc_mask, rho=rho_val)
remainer_mask = trn_mask.detach().clone()
trn_mask, val_mask = trn_mask.unsqueeze(0), val_mask.unsqueeze(0)

all_trn_mask, loss_mask = torch.empty((num_reps, H, W,n_coil,N_time)), torch.empty((num_reps, H, W,n_coil,N_time))  #(25,192,158,32,5)
for jj in range(num_reps):
    all_trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_data,remainer_mask,rho=rho_train)  # mask:(25,640,368)

#..............................put all mask to device..................................
trn_mask,val_mask,all_trn_mask,loss_mask,acc_mask = trn_mask.to(device),val_mask.to(device),all_trn_mask.to(device),loss_mask.to(device),acc_mask.to(device)
kspace_data,sensitivity_maps = kspace_data.to(device),sensitivity_maps.to(device)

#3. Mask all data
#..............................creating initial M,T2 data..................................
val_input,trn_input = torch.ones((1,H,W,3),dtype=torch.float32).to(device),torch.ones((num_reps, H, W, 3), dtype=torch.float32).to(device)

#..............................creating initial kspace data..................................
val_ref_kspace = torch.empty(kspace_data.shape,dtype=torch.complex64).to(device)  #(192,158,5,32)
val_input_k = torch.empty(kspace_data.shape,dtype=torch.complex64).to(device)     #(192,158,5,32)

trn_ref_kspace = torch.empty((num_reps, H, W, N_time, n_coil),dtype=torch.complex64).to(device)  # ref_kspace:(25,192,158,5,32),Loss
trn_input_k = torch.empty((num_reps, H, W, N_time, n_coil), dtype=torch.complex64).to(device)  # (25,192,158,5,32)

for i in range(N_time):
    val_input_k[...,i,:] = kspace_data[...,i,:] * trn_mask[0,...,i]
    val_ref_kspace[...,i,:] = kspace_data[...,i,:] * val_mask[0,...,i]

for jj in range(num_reps):
    for i in range(N_time):
        trn_input_k[jj, ..., i, :] = kspace_data[..., i, :] * all_trn_mask[jj,...,i]
        trn_ref_kspace[jj, ..., i, :] = kspace_data[..., i, :] * loss_mask[jj,...,i]

#3. Pretrain data
# %% Prepare the data for the training
sensitivity_maps = torch.tile(sensitivity_maps.unsqueeze(0),(num_reps,1,1,1))                 #(25,192,158,32)
trn_ref_kspace = torch.stack([trn_ref_kspace.real,trn_ref_kspace.imag],dim=-1)          #ref_kspace:(25,192,158,5,32,2)
val_ref_kspace = torch.stack([val_ref_kspace.real,val_ref_kspace.imag],dim=-1).unsqueeze(0)  #(1,192,158,5,32,2)
val_input_k = val_input_k.unsqueeze(0)            # (1,25,256,208,5,32)

#Pretrain to get a raw data
trn_out = torch.empty(trn_input.shape,dtype=torch.float32).cuda()
val_out = torch.empty(val_input.shape,dtype=torch.float32).cuda()

for i in range(num_reps):
      trn_out[i] = dc.NLCG(kimage_Data=trn_input_k[i],sens_map=sensitivity_maps[i],mask=all_trn_mask[i],M=trn_input[i],z=None,iterations=pretrain_iter,lam=0.0)

val_out = dc.NLCG(kimage_Data=val_input_k[0],sens_map=sensitivity_maps[0],mask=trn_mask[0],M=val_input[0],z=None,iterations=pretrain_iter,lam=0.0).unsqueeze(0)

directory = os.path.join('T1 mapping pretrain data', str(acc_rate)+'Rate' + '_' + str(pretrain_iter) + 'Pretrain_iter')
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

#os.system('shutdown')
"""
4. Apply brain mask to block surrounding downsampled area, if necessary
#Apply brain mask to block surrounding area in image space, then convert this new data to kspace as ref_kspace
def mask_surrounding_kspace(input_kspace,sensitivity_maps,brain_mask,device='cuda'):

    #args: input_kspace: data that wait to be masked with brain mask, shape: (num_re_reps,H,W,N_time,n_coil,2)
    #sensitivity_maps: (25,256,208,12)
    #brain_mask: (1,1,1,256,208,1)
    #return: kspace with brain mask, which can be used to calculate loss in Neural Network. (N_reps,H,W,N_time,n_coil,2)
    brain_mask = brain_mask.to(torch.complex64)
    print(233)
    print(input_kspace.shape,sensitivity_maps.shape,brain_mask.shape)
    
    input_kspace,sensitivity_maps,brain_mask = input_kspace.cpu(),sensitivity_maps.cpu(),brain_mask.cpu()
    num_reps,H,W,N_time,n_coil,_ = input_kspace.shape
    
    input_kspace = input_kspace.permute(0,3,4,1,2,5)        #(25,31,12,256,208,2)
    input_img = fastmri.ifft2c(input_kspace)
    input_img_comp = torch.complex(input_img[...,0],input_img[...,1])    #(25,31,12,256,208)
    c_input_img_comp = input_img_comp * torch.conj(sensitivity_maps.unsqueeze(1).permute(0,1,4,2,3))   #(25,31,12,256,208)
    sum_img_comp = torch.empty((num_reps,N_time,H,W),dtype=torch.complex64)                 #(25,31,256,208)
    
    for i in range(N_time):
        sum_img_comp[:,i,...] = torch.sum(c_input_img_comp[:,i],dim=1)
    sum_img_comp = sum_img_comp * brain_mask[0,...,0]

    coil_img_ref = torch.empty((num_reps,H,W,N_time,n_coil),dtype=torch.complex64)   #(25,256,208,31,12)
    for i in range(N_time):        #25,256,208,12          25,256,208,1                     25,12,256,208
        coil_img_ref[:,:,:,i,:] = sum_img_comp[:,i].unsqueeze(-1) * sensitivity_maps
    coil_img_ref = torch.stack([coil_img_ref.real,coil_img_ref.imag],dim=-1).permute(0,3,4,1,2,5)     #(25,31,12,256,208,2)
    mask_ref_kspace = fastmri.fft2c(coil_img_ref).permute(0,3,4,1,2,5)

    return mask_ref_kspace.to(device)

trn_ref_kspace, val_ref_kspace = mask_surrounding_kspace(trn_ref_kspace,sensitivity_maps,brain_mask,device), mask_surrounding_kspace(val_ref_kspace,sensitivity_maps[0].unsqueeze(0),brain_mask,device)

for i in range(trn_input.shape[0]):
    trn_input[i] = trn_input[i] * brain_mask.squeeze().unsqueeze(-1)
for i in range(val_input.shape[0]):
    val_input[i] = val_input[i] * brain_mask.squeeze().unsqueeze(-1)
"""