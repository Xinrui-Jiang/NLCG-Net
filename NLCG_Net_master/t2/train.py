import os
import time
import numpy as np
import scipy.io as scio
import torch
import fastmri
from torch.utils.data import DataLoader
import utils, UnrollNet
import Unroll_simplenet

from modules import RMSELoss,MixL1L2Loss, Dataset, Dataset_Inference, train, validation, test
import matplotlib.pyplot as plt
import data_consistency as dc

#0. Set hyperparameter for model
num_reps = 25
batchSize = 1
#selected_list = list(range(1,32,4))
selected_list = [1,3,5,7,9,11,13,15]

pretrain_iter,CG_iter,nb_unroll_blocks,nb_res_blocks = 800,20,3,2     #一个unroll block要5308mb

acc_rate = 6
learning_rate = 1e-4
epochs, stop_training = 300, 25
ep, val_loss_tracker = 0, 0
device = 'cuda'

#1. Load sensitivity maps
sensitivity_maps = torch.from_numpy(scio.loadmat(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/coil_sensitivities.mat")['sens']).to(torch.complex64)
sensitivity_maps = torch.tile(sensitivity_maps.unsqueeze(0),(num_reps,1,1,1))                 #(25,256,208,12), block, if it has been applied formerly
sensitivity_maps = sensitivity_maps.to('cpu')

#2. Load criterion reference
ref_r2 = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/TE<200result/800no mask/recon_result.pth")[...,2].to(device)
brain_mask = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/2to32,8data/csf_block_mask.pth").to(torch.float32).to(device)
trn_nrmse_record,val_nrmse_record = [], []
trn_imgnrmse_record,val_imgnrmse_record = [], []
print(brain_mask.shape)

#3. Load acceleration mask
#acc_mask = torch.from_numpy(scio.loadmat(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/ACC6, masks/ACC6, no ACS.mat")['mask_all']).to('cpu')    #(256,208,12,8)
acc_mask = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/ACC6 uniform,24,93-116 are 1.pth").to('cpu')    #(256,208,12,8).to(torch.float32)


#4. Load pretrain data, which are masks, data after 800 iteration for initialization
data_directory = r"/root/autodl-tmp/MOBA-T2mapping-master/TE<200,6acc_93-116_24pretrain_data/6Rate_800Pretrain_iter/"

trn_input = torch.load(os.path.join(data_directory,"trn_out.pth")).to('cpu')          #(25,256,208,3)
trn_input_k = torch.load(os.path.join(data_directory,"trn_input_k.pth")).to('cpu')
all_trn_mask = torch.load(os.path.join(data_directory,"all_trn_mask.pth")).to('cpu')
loss_mask = torch.load(os.path.join(data_directory,"loss_mask.pth")).to('cpu')
trn_ref_kspace = torch.load(os.path.join(data_directory,"trn_ref_kspace.pth")).to('cpu')

val_input = torch.load(os.path.join(data_directory,"val_out.pth")).to('cpu')
val_input_k = torch.load(os.path.join(data_directory,"val_input_k.pth")).to('cpu')
trn_mask = torch.load(os.path.join(data_directory,"trn_mask.pth")).to('cpu')
val_mask = torch.load(os.path.join(data_directory,"val_mask.pth")).to('cpu')
val_ref_kspace = torch.load(os.path.join(data_directory,"val_ref_kspace.pth")).to('cpu')

#5. Set random seed
torch.manual_seed(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#6. Create dataset and dataloader
train_data = Dataset(trn_input,trn_input_k,all_trn_mask, loss_mask, sensitivity_maps, trn_ref_kspace)
train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)       #长度25
val_data = Dataset(val_input,val_input_k,trn_mask, val_mask,sensitivity_maps[0].unsqueeze(0), val_ref_kspace)
val_loader = DataLoader(val_data, batch_size=batchSize, shuffle=False)


#7. Define models, loss function and optimizers
model_directory = os.path.join('TE<200,correctUnet,6acc,acs', str(acc_rate)+\
                         '_' + str(pretrain_iter) + '1e-4'+ str(learning_rate) + 'LR'+str(CG_iter) + 'CG iter'+ str(nb_unroll_blocks) + 'Unrolls'+str(nb_res_blocks)+'Unet' )
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

#model = Unroll_simplenet.UnrolledNet(batchSize=batchSize,CG_iter=CG_iter,nb_unroll_blocks=nb_unroll_blocks,device=device).to(device)
model = UnrollNet.UnrolledNet(batchSize=batchSize,CG_iter=CG_iter,nb_unroll_blocks=nb_unroll_blocks,device=device).to(device)

loss_fn = MixL1L2Loss()
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

"""
#Optional: Load chectpoint of previous model
best_checkpoint = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/TE<200,Resnet,6acc,acs/6_800every scale,OG setting,Pt_iter0.0001LR20CG iter3Unrolls2Unet/best.pth")
model.load_state_dict(best_checkpoint["model_state"])
"""
#8. Perform training
total_train_loss, total_val_loss = [], []
valid_loss_min = np.inf

start_time = time.time()
while ep < epochs and val_loss_tracker < stop_training:

    tic = time.time()
    trn_loss, lamdas,trn_nrmse,trn_imgnrmse = train(train_loader, model, loss_fn, optimizer, ep,ref_r2,brain_mask,device=device)
    total_train_loss.append(trn_loss)
    trn_nrmse_record.append(trn_nrmse)
    trn_imgnrmse_record.append(trn_imgnrmse)
    
    val_loss,val_nrmse,val_imgnrmse = validation(val_loader, model, loss_fn,ref_r2,brain_mask, device=device)
    total_val_loss.append(val_loss)
    val_nrmse_record.append(val_nrmse)
    val_imgnrmse_record.append(val_imgnrmse)
    
    print('epoch: {}, average trn_loss: {}, val loss: {}'.format(ep,trn_loss,val_loss))

    # save the best checkpoint
    checkpoint = {
        "epoch": ep,
        "valid_loss_min": val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    if val_loss <= valid_loss_min:
        valid_loss_min = val_loss
        torch.save(checkpoint, os.path.join(model_directory, "best.pth"))
        val_loss_tracker = 0  # reset the val loss tracker each time a new lowest val error is achieved
    else:
        val_loss_tracker += 1

    toc = time.time() - tic
    print("Epoch:", ep + 1, ", elapsed_time = ""{:f}".format(toc), ", trn loss = ", "{:.3f}".format(trn_loss),", val loss = ", "{:.3f}".format(val_loss))
    
    scio.savemat(os.path.join(model_directory, 'later TrainingLog,{},{},{}.mat'.format(CG_iter,nb_unroll_blocks,nb_res_blocks)), {'trn_loss': total_train_loss, 'val_loss': total_val_loss,'trn_nrmse':trn_nrmse_record,'val_nrmse':val_nrmse_record,'trn_imgnrmse':trn_imgnrmse_record,'val_imgnrmse':val_imgnrmse_record})
    ep += 1

end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ', ((end_time - start_time) / 3600), ' hours')
torch.save(model,os.path.join(model_directory,'zs model.pth'))

#9. Inference the whole image

kspace_data = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/rotated_kdata.pth")[:,:,selected_list,:].to(torch.complex64).to(device)

H,W,N_time,n_coil = kspace_data.shape
test_mask = acc_mask.cuda()
inference_input_k = (kspace_data * test_mask.permute(0,1,3,2)).unsqueeze(0)

inference_input = torch.ones((1,H,W,3),dtype=torch.float32).cuda()            #(1,256,208,3)
inference_input[0] = dc.NLCG(kimage_Data=inference_input_k[0],sens_map=sensitivity_maps[0].cuda(),mask=test_mask,M=inference_input[0],z=None,iterations=pretrain_iter,lam=0.)

test_data = Dataset_Inference(inference_input,inference_input_k,test_mask.unsqueeze(0),test_mask.unsqueeze(0), sensitivity_maps[0].unsqueeze(0))
test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

# load the best checkpoint, and perform reconstruction
best_checkpoint = torch.load(os.path.join(model_directory,'best.pth'))
model.load_state_dict(best_checkpoint["model_state"])
zs_ssl_recon = test(test_loader, model, device)

torch.save(zs_ssl_recon,os.path.join(model_directory,'recon result.pth'))
os.system('shutdown')