import torch
import torch.nn as nn
import time
import parser_ops
import os

parser = parser_ops.get_parser()
args = parser.parse_args([])
ref_data = torch.load(os.path.join(args.ref_data_dir,args.ref_img_name)).to(args.device) / args.k_scaling_factor

class Dataset(torch.utils.data.Dataset):
    def __init__(self,trn_M, trn_kspace, trn_mask, loss_mask, sens_maps, ref_kspace):
        self.trn_M = trn_M
        self.trn_K = trn_kspace
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.sens_maps = sens_maps
        self.ref_kspace = ref_kspace
        
    def __len__(self):
        return len(self.trn_M)
        
    def __getitem__(self,idx):
        input_M = self.trn_M[idx]
        input_K = self.trn_K[idx]
        trn_mask , loss_mask = self.trn_mask[idx], self.loss_mask[idx]
        sens_maps =  self.sens_maps[idx]
        ref_kspace = self.ref_kspace[idx]

        return input_M, input_K, trn_mask, loss_mask, sens_maps, ref_kspace

class Dataset_Inference(torch.utils.data.Dataset):
    def __init__(self,trn_M, trn_kspace, test_mask, loss_mask, sens_maps):
        self.trn_M = trn_M
        self.trn_K = trn_kspace
        self.test_mask = test_mask
        self.loss_mask = loss_mask
        self.sens_maps = sens_maps

        
    def __len__(self):
        return len(self.trn_M)
        
    def __getitem__(self,idx):
        input_M = self.trn_M[idx]
        input_K =self.trn_K[idx]
        test_mask, loss_mask = self.test_mask[idx], self.loss_mask[idx]
        sens_maps =  self.sens_maps[idx]

        return input_M, input_K, test_mask, loss_mask, sens_maps


class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6,scalar=1/2):
        super().__init__()
        #self.mse = nn.MSELoss()
        self.eps = eps
        self.scalar=scalar
    def forward(self, yhat, y):
        #y: (1,256,208,8,12,2)
        loss = self.scalar*torch.sum(torch.linalg.vector_norm(yhat-y,dim=(0,1,2,4,5)) / torch.linalg.vector_norm(y,dim=(0,1,2,4,5))) + self.scalar*torch.sum(torch.linalg.vector_norm(yhat-y,ord=1,dim=(0,1,2,4,5)) / torch.linalg.vector_norm(y, ord=1,dim=(0,1,2,4,5)))
        
        return loss
        
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yhat,y):
        #input shape: (1,256,208,8,12,2)
        loss = torch.sum(torch.linalg.vector_norm(yhat-y,dim=(0,1,2,4,5)) / torch.linalg.vector_norm(y,dim=(0,1,2,4,5)))
        return loss

def NRMSE(est,ref_est,mask):
    #only calculate R2 NRMSE, rather than T2
    nrmse = torch.norm(est[mask!=0]-ref_est[mask!=0]) / torch.norm(ref_est[mask!=0])
    return nrmse

def imgNRMSE(result,ref_img,mask):
    """
    #calculate mean NRMSE over 31 images, or 8 images
    #ref_img: (256,208,8)
    #mask: block surrounding area, (256,208)
    #result: (256,208,3) or (256,208,2)
    """
    mask = mask.unsqueeze(-1)
    img = torch.empty(ref_img.shape,dtype=ref_img.dtype).cuda()
    num = img.shape[-1]
    te_list = args.echo_time_list
    
    M = torch.complex(result[...,0],result[...,1])
    for i,te in enumerate(te_list):
        img[...,i] = torch.abs(torch.exp(-te*result[...,-1]) * M)

    imgnrmse = torch.norm((img-ref_img)*mask,dim=(0,1)) / torch.norm(ref_img*mask,dim=(0,1))
    avg_nrmse = torch.sum(imgnrmse) / num
    return avg_nrmse  
    

def train(train_loader, model, loss_fn, optimizer, ep,ref_r2,brain_mask, device = torch.device('cuda')):
    avg_trn_cost = 0
    total_nrmse = 0
    total_imgnrmse = 0
    model.train()
    for ii,batch in enumerate(train_loader):
        print('currently {} epoch, {}th data'.format(ep,ii))
        input_M,input_k,trn_mask,loss_mask,sens_maps,ref_kspace = batch
        input_M,input_k,trn_mask,loss_mask,sens_maps,ref_kspace = input_M.to(device),input_k.to(device),trn_mask.to(device),loss_mask.to(device),sens_maps.to(device),ref_kspace.to(device)
        """Forward Path"""
        nw_img_output, lamdas,nw_kspace_output = model(input_k,input_M,trn_mask,loss_mask,sens_maps)
        
        print('current lam: {}'.format(lamdas))
        
        """Loss"""
        trn_loss = loss_fn(nw_kspace_output,ref_kspace)
        
        """Backpropagation"""
        optimizer.zero_grad()
        trn_loss.backward()
        optimizer.step()

        avg_trn_cost += trn_loss.item() /  len(train_loader)
        with torch.no_grad():
            total_nrmse += NRMSE(nw_img_output[0,...,2],ref_r2,brain_mask)
            total_imgnrmse += imgNRMSE(nw_img_output[0],ref_data,brain_mask)

    avg_nrmse,avg_imgnrmse = total_nrmse / args.num_reps, total_imgnrmse / args.num_reps
    print('trn_avg_nrmse: {}, img_nrmse: {}'.format(avg_nrmse,avg_imgnrmse))
    return avg_trn_cost, lamdas,avg_nrmse.cpu(),avg_imgnrmse.cpu()

def validation(val_loader, model, loss_fn,ref_r2,brain_mask, device = torch.device('cuda')):
    avg_val_cost = 0
    avg_nrmse,avg_imgnrmse = 0,0
    model.eval()
    with torch.no_grad():
        for ii,batch in enumerate(val_loader):
            input_M,input_k,trn_mask,loss_mask,sens_maps,ref_kspace= batch
            input_M,input_k,trn_mask,loss_mask,sens_maps,ref_kspace = \
                input_M.to(device), input_k.to(device),trn_mask.to(device), loss_mask.to(device), sens_maps.to(device), ref_kspace.to(device)
            """Forward Path"""
            nw_img_output, lamdas,nw_kspace_output = model(input_k,input_M,trn_mask,loss_mask,sens_maps)
            
            """Loss"""
            val_loss =loss_fn(nw_kspace_output,ref_kspace)
            
            avg_val_cost += val_loss.item() / len(val_loader)
            avg_nrmse += NRMSE(nw_img_output[0,...,2],ref_r2,brain_mask) / len(val_loader)
            avg_imgnrmse += imgNRMSE(nw_img_output[0],ref_data,brain_mask) / len(val_loader)
          
    print('val_avg_nrmse: {}, img_nrmse: {}'.format(avg_nrmse,avg_imgnrmse))
    return avg_val_cost,avg_nrmse.cpu(),avg_imgnrmse.cpu()

def test(test_loader, model, device = torch.device('cuda')):
    model.eval()
    with torch.no_grad():
        for ii,batch in enumerate(test_loader):
            input_M, input_k, trn_mask, loss_mask, sens_maps = batch
            input_M, input_k, trn_mask, loss_mask, sens_maps = \
                input_M.to(device), input_k.to(device), trn_mask.to(device), loss_mask.to(device), sens_maps.to(device)

            """ Forward Path """
            nw_img_output, lamdas,nw_kspace_output = model(input_k,input_M,trn_mask,loss_mask,sens_maps)

    return nw_img_output
