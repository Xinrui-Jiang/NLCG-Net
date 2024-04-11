import os
import time
import numpy as np
import scipy.io as scio
import torch
import parser_ops
from torch.utils.data import DataLoader
import utils, UnrollNet
from modules import RMSELoss,MixL1L2Loss, Dataset, Dataset_Inference, train, validation, test
import data_consistency as dc

parser = parser_ops.get_parser()
args = parser.parse_args([])

#0. Fix random seed
torch.manual_seed(3407)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('current echo time: {}, r_scaling_factor: {}'.format(args.echo_time_list,args.r_scaling_factor))

#2. Load ralevant data
kspace_data = torch.load(os.path.join(args.data_dir, args.kdata_name)).to(torch.complex64).to(args.device) #(256,208,8,12)    #这里也要改
kspace_data = kspace_data / args.k_scaling_factor
sensitivity_maps = torch.from_numpy(scio.loadmat(os.path.join(args.data_dir,args.sens_name))['sens']).to(torch.complex64)
sensitivity_maps = torch.tile(sensitivity_maps.unsqueeze(0),(args.num_reps,1,1,1))              #(25,256,208,12)
acc_mask = torch.load(os.path.join(args.data_dir,"confirmed_t2_ACC{},noACS.pth".format(args.acc_rate)))    #(256,208,12,8)
                      
#3. Load criterion reference
ref_r2 = torch.load(os.path.join(args.ref_data_dir,args.ref_r_name))[...,2].to(args.device) / args.r_scaling_factor
brain_mask = torch.load(os.path.join(args.ref_data_dir,args.csf_mask_name)).to(torch.float32).to(args.device)
trn_nrmse_record,val_nrmse_record,trn_imgnrmse_record,val_imgnrmse_record = [], [], [], []
print(brain_mask.shape)

#4. Load pretrain data, which are masks, data after several iteration for initialization
trn_input_k = torch.load(os.path.join(args.SSLdata_dir,"trn_input_k.pth")).to('cpu')     #(25,256,208,3)
all_trn_mask = torch.load(os.path.join(args.SSLdata_dir,"all_trn_mask.pth")).to('cpu')
loss_mask = torch.load(os.path.join(args.SSLdata_dir,"loss_mask.pth")).to('cpu')
trn_ref_kspace = torch.load(os.path.join(args.SSLdata_dir,"trn_ref_kspace.pth")).to('cpu')
val_input_k = torch.load(os.path.join(args.SSLdata_dir,"val_input_k.pth")).to('cpu')
trn_mask = torch.load(os.path.join(args.SSLdata_dir,"trn_mask.pth")).to('cpu')
val_mask = torch.load(os.path.join(args.SSLdata_dir,"val_mask.pth")).to('cpu')
val_ref_kspace = torch.load(os.path.join(args.SSLdata_dir,"val_ref_kspace.pth")).to('cpu')

init_guess = torch.load(os.path.join(args.SSLdata_dir,args.initial_guess_name)).to(torch.float32).to(args.device)
trn_input,val_input = init_guess.detach().clone().unsqueeze(0).repeat(args.num_reps,1,1,1),init_guess.detach().clone().unsqueeze(0).repeat(1,1,1,1)

#5. Create dataset and dataloader
train_data = Dataset(trn_input,trn_input_k,all_trn_mask, loss_mask, sensitivity_maps, trn_ref_kspace)
train_loader = DataLoader(train_data, batch_size=args.batchSize, shuffle=True)       #长度25
val_data = Dataset(val_input,val_input_k,trn_mask, val_mask,sensitivity_maps[0].unsqueeze(0), val_ref_kspace)
val_loader = DataLoader(val_data, batch_size=args.batchSize, shuffle=False)


#6. Define models, loss function and optimizers
model_directory = os.path.join('k_{}_r_{}_{}'.format(args.k_scaling_factor, args.r_scaling_factor,args.data_opt), 'ACC'+str(args.acc_rate)+'_LR' +str(args.learning_rate)+str(args.CG_Iter) + 'CG iter'+ str(args.nb_unroll_blocks) + 'Unrolls'+str(args.nb_res_blocks)+'resnet' )
if not os.path.exists(model_directory):
    os.makedirs(model_directory)


model = UnrollNet.UnrolledNet(batchSize=args.batchSize,CG_iter=args.CG_Iter,nb_unroll_blocks=args.nb_unroll_blocks,device=args.device).to(args.device)
loss_fn = MixL1L2Loss()
optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)

"""
#Optional: Load chectpoint of previous model
best_checkpoint = torch.load(r"../best.pth")
model.load_state_dict(best_checkpoint["model_state"])
"""
#8. Perform training
ep, val_loss_tracker = 0, 0
total_train_loss, total_val_loss = [], []
valid_loss_min = np.inf

start_time = time.time()
while ep < args.epochs and val_loss_tracker < args.stop_training:

    tic = time.time()
    trn_loss, lamdas,trn_nrmse,trn_imgnrmse = train(train_loader, model, loss_fn, optimizer, ep,ref_r2,brain_mask,device=args.device)
    total_train_loss.append(trn_loss)
    trn_nrmse_record.append(trn_nrmse)
    trn_imgnrmse_record.append(trn_imgnrmse)
    
    val_loss,val_nrmse,val_imgnrmse = validation(val_loader, model, loss_fn,ref_r2,brain_mask, device=args.device)
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
    
    scio.savemat(os.path.join(model_directory, 'later TrainingLog,{},{},{}.mat'.format(args.CG_Iter,args.nb_unroll_blocks,args.nb_res_blocks)), {'trn_loss': total_train_loss, 'val_loss': total_val_loss,'trn_nrmse':trn_nrmse_record,'val_nrmse':val_nrmse_record,'trn_imgnrmse':trn_imgnrmse_record,'val_imgnrmse':val_imgnrmse_record})
    ep += 1

end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ', ((end_time - start_time) / 3600), ' hours')
torch.save(model,os.path.join(model_directory,'zs model.pth'))

#9. Inference the whole image
test_mask = acc_mask.to(args.device)
inference_input_k = (kspace_data * test_mask.permute(0,1,3,2)).unsqueeze(0)

inference_input = init_guess.detach().clone().unsqueeze(0).repeat(1,1,1,1).to(args.device)  #(1,256,208,3)
test_data = Dataset_Inference(inference_input,inference_input_k,test_mask.unsqueeze(0),test_mask.unsqueeze(0), sensitivity_maps[0].unsqueeze(0))
test_loader = DataLoader(test_data, batch_size=args.batchSize, shuffle=False)

# load the best checkpoint, and perform reconstruction
best_checkpoint = torch.load(os.path.join(model_directory,'best.pth'))
model.load_state_dict(best_checkpoint["model_state"])
zs_ssl_recon = test(test_loader, model, args.device)

torch.save(zs_ssl_recon,os.path.join(model_directory,'recon result.pth'))
os.system('shutdown')