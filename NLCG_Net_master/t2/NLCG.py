import torch
import numpy as np
import math
import fastmri
import time
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio

def abnormal_detector(x,variable_name):
    if torch.any(x.isnan()) == True:
        print(variable_name + ' is a nan!')
        time.sleep(2e8)
    return 0


def NRMSE(est,ref_est,mask,Trange=200):
    clip_est = torch.clip(est,0,Trange)
    clip_ref_est = torch.clip(ref_est,0,Trange)
    nrmse = torch.norm(clip_est[mask!=0]-clip_ref_est[mask!=0]) / torch.norm(clip_ref_est[mask!=0])
    return nrmse

def imgNRMSE(result,ref_img,mask,TE=0.0115,varpro=False,possible_te_range= range(1,32,4)):
    """
    #calculate mean NRMSE over 31 images, or 8 images
    #ref_img: (256,208,31)
    #mask: (256,208)
    #result: (256,208,3) or (256,208,2)
    """
    mask = mask.unsqueeze(-1)
    params_num = result.shape[-1]
    img = torch.empty(ref_img.shape,dtype=ref_img.dtype).cuda()
    num = img.shape[-1]
    possible_te_tange = [1,3,5,7,9,11,13,15]
    
    if params_num == 3:
        M = torch.abs(torch.complex(result[...,0],result[...,1]))
    else:
        M = result[...,0]
    for i,j in enumerate(possible_te_range):
        te = (j + 1) * TE
        if varpro==True:
            te = te * 1000
            img[...,i] = torch.exp(-te/result[...,-1]) * M
        else:
            img[...,i] = torch.exp(-te*result[...,-1]) * M 

    imgnrmse = torch.norm((img-ref_img)*mask,dim=(0,1)) / torch.norm(ref_img*mask,dim=(0,1))
    all_nrmse = torch.sum(imgnrmse) / num
    return all_nrmse  

class operator():
    def __init__(self, kimage_Data, sens_map, mask, z, lam=0.001, TE=0.0115, criteria=5e-6):
        self.kimages = kimage_Data  # (256,208,31,12), 31 received image-space images constructed by rss
        self.z = z
        self.mask = mask  # (256,208)
        self.H, self.W, self.N = kimage_Data.shape[0], kimage_Data.shape[1], kimage_Data.shape[2]
        self.lam = lam
        self.sens = sens_map  # (256,208,12)
        self.n_coils = self.sens.shape[2]
        self.criteria = criteria
        #self.eps = torch.finfo(torch.float32).eps
        self.eps = 1e-6
        if self.N < 31:
            #self.te_list = [0.023,0.069,0.115,0.161,0.207,0.253,0.299,0.345]
            self.te_list = [0.023,0.046,0.069,0.092,0.115,0.138,0.161,0.184]

    def PFCM_op(self, input_x):
        """
        :param input_x: (256,208,3), (256,208,0) is Mx,(256,208,1) is My, (256,208,2) is v2 = 1/T2
        :return:  PFCM(X),        (12*31,256,208)
        """
        coil_img = torch.empty((self.N * self.n_coils, self.H, self.W),
                               dtype=torch.complex64).cuda()  # (12*31,256,208),coil first
                               
        M_complex = torch.complex(input_x[..., 0], input_x[..., 1])  # (256,208)

        for i,j in enumerate(self.te_list):
            coil_img[i * self.n_coils:(i + 1) * self.n_coils, ...] = M_complex * torch.exp(-j * input_x[..., 2])

        coil_img = coil_img * torch.tile(self.sens.permute(2, 0, 1), (self.N, 1, 1))
        coil_img = torch.stack([coil_img.real, coil_img.imag], dim=-1)  # (12*31,256,208,2)

        kspace = fastmri.fft2c(coil_img)
        kspace = torch.complex(kspace[..., 0], kspace[..., 1])  # (12*31,256,208)

        masked_kspace = self.mask * kspace

        return masked_kspace

    def CFH_op(self, x_y):
        """
        :param x_y: (12*31,256,208), PFCM(X) - Y
        :return: CFH(target),      (31,256,208)
        """
        image_space_coil_imgs = fastmri.ifft2c(torch.stack([x_y.real, x_y.imag], dim=-1))
        image_space_coil_imgs = torch.complex(image_space_coil_imgs[..., 0],
                                              image_space_coil_imgs[..., 1])  # (12*31,256,208)

        image_space_comb = image_space_coil_imgs * torch.tile(torch.conj(self.sens.permute(2, 0, 1)),
                                                              (self.N, 1, 1))  # (12*31,256,208)

        sum_image_space_comb = torch.empty((self.N, self.H, self.W), dtype=torch.complex64).cuda()  # (31,256,208)
        for i in range(self.N):
            sum_image_space_comb[i, :, :] = torch.sum(image_space_comb[i * self.n_coils:(i + 1) * self.n_coils, :, :],
                                                      dim=0)

        return sum_image_space_comb

    def grad_M(self, input_x):
        """
        :param input_x: (256,208,3), (256,208,0) is Mx,(256,208,1) is My, (256,208,2) is v2 = 1/T2
        :return: (e^(-t*v2), -tM0e^(-t*v2), (256,208,31*2,3)         #real/complex first, coil later, echo last
        """
        gradM = torch.zeros((self.H, self.W, self.N, 2, 3), dtype=torch.float32).cuda()

        for i,j in enumerate(self.te_list):
            gradM[:, :, i, 0, 0] = torch.exp(-j * input_x[:, :, 2])  # gradient of M(X) to M0
            gradM[:, :, i, 1, 1] = torch.exp(-j * input_x[:, :, 2])
            gradM[:, :, i, 0, 2] = -j * input_x[:, :, 0] * torch.exp(-j * input_x[:, :, 2])
            gradM[:, :, i, 1, 2] = -j * input_x[:, :, 1] * torch.exp(-j * input_x[:, :, 2])

        gradM = gradM.contiguous().view(self.H, self.W, -1, 3)

        return gradM

    def grad_F(self, input_x):

        masked_kspace = self.PFCM_op(input_x)

        x_y = masked_kspace - self.kimages.clone().contiguous().view(self.H, self.W, -1).permute(2, 0, 1)  # (12*31,256,208)
        sum_image_space_comb = self.CFH_op(x_y).permute(1, 2, 0)  # (256,208,31)

        gradM = self.grad_M(input_x).permute(0, 1, 3, 2)  # (256,208,3,31*2)
        image_space_comb = torch.stack([sum_image_space_comb.real, sum_image_space_comb.imag], dim=-1)  # (256,208,31,2)
        image_space_comb = image_space_comb.contiguous().view(self.H, self.W, -1)  # (256,208,31*2)

        gradF = torch.matmul(gradM, image_space_comb.unsqueeze(-1)).squeeze()
        
        if self.z != None:
            gradF = gradF + self.lam * (input_x - self.z)
        
        return gradF

    def direction_projector(self,input_x,a,p):
        #to avoid R2 becoming negative, make projection and accordingly adjust direction p
        projected_p = p
        if a == 0:
            return projected_p
          
        step = input_x + a * p
        negative_r2_place = step[...,2] < 0
        
        if torch.any(negative_r2_place) == True:
            print('attention, projection is being operated')
            projected_p[...,2][negative_r2_place] = -input_x[...,2][negative_r2_place] / a
            
            projected_p[projected_p.isnan()] = 0
            projected_p[projected_p.isinf()] = 0
         
        return projected_p
    
    def dFd_alpha(self,input_x,alpha,p):        # calculate dF/d alpha, following chain rule

        new_step = input_x + alpha * p
        dFdx = self.grad_F(new_step)            #(256,208,3)
        dFda = torch.sum(dFdx * p)  # (256,208,1,1)
        
        return dFda   
            
    def secant_method(self, input_x, p):
        """
        apply Line Search to find the curent best step size alpha, which suffice:
        argmin F(x + alpha * p), i.e., dF/d alpha = 0. 
        as long as dF/d alpha is close to a small number, we can claim that we have got a correct alpha
        :param input_x: (256,208,3), (256,208,0) is Mx,(256,208,1) is My, (256,208,2) is v2 = 1/T2
        :param p: current step direction, (256,208,3), (256,208,0) is Mx direction, (256,208,2) is T2 direction
        :param z: regularization term z, (256,208,3)
        :return: alpha, current best step size, (256,208,3)
        """
      
        a0 = 0.
        a1 = 1.

        p_normalize = p

        error = self.criteria + 1
        count = 0

        while error > self.criteria:
            with torch.no_grad():
                p1 = self.direction_projector(input_x,a1,p_normalize)
                p0 = self.direction_projector(input_x,a0,p_normalize)
                
                d1 = self.dFd_alpha(input_x, a1, p1)
                d0 = self.dFd_alpha(input_x, a0, p0)
                ddy = (d1 - d0) / ( (a1 - a0)+self.eps )
                
                a_new = a1 - d1 / (ddy+self.eps)
            p_new = self.direction_projector(input_x,a_new,p_normalize)
            
            error = torch.abs(self.dFd_alpha(input_x, a_new, p_new))
            #print('count: {}, error: {}, ddy: {}, d1: {}, d0: {}, a1: {},a0: {}'.format(count, error,ddy,d1,d0,a1,a0))

            a0 = a1
            a1 = a_new
            count += 1

            if (torch.abs(a1-a0) < 1e-3) or (count > 10 and error < 1e-2):
                break
            if count > 20:
                break
                
        #print('choose a: {}'.format(a1))
        #print('choose a: {}, step max: {}, step min: {}, step mean: {}'.format(a1,torch.max(a1 * p_normalize), torch.min(a1 * p_normalize),torch.mean(a1 * p_normalize)))
        best_step = a_new * p_new

        return best_step

def NLCG(kimage_Data, sens_map, mask, M, z=None, iterations=300, lam=0.001, criteria=5e-6,TE=0.0115):
    """
    #perform non-linear conjugate gradient method to reconstruct (M,T2) from 31 images constructed by rss
    :param image_Data: 31 images reconstructed by RSS, (256,208,31)
    :param sens_map: 12 sensitivity maps, (256,208,12)
    :param mask: Accelerate map, (256,208)
    :param M: Corresponding M,R2 to be estimated, can be all one(initialize) or from last NL-CG block output,(256,208,3)
    :param z: Last regularizer output
    :param iterations: NL-CG iteration number
    :param lam: Regularizartion parameter
    :param criteria: Stopping criteria used in secant methods
    :param condition: Choose RF or Daiyuan method in NL-CG
    :param TE: Interleave time, using 's' as unit
    :projection: Decide if performing projection to let all R2_hat be non-negative
    :return: Reconstructed (M,T2), (256,208,3)
    """
    varpro_t2 = torch.from_numpy(scio.loadmat(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/2to32,8data/0-1000 2to32 8 point T2.mat")['T2std']).cuda()
    brain_mask = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/2to32,8data/csf_block_mask.pth").cuda()
    #selected_list = list(range(1,32,4))
    selected_list = [1,3,5,7,9,11,13,15]
    ref_img = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/reconstructed_img.pth")[:,:,selected_list].cuda()

    r_record,R_record,nrmse_record,mse_record = [],[],[],[]
    
    encoder = operator(kimage_Data, sens_map, mask, z, lam, TE, criteria)

    t = 0
    p = -encoder.grad_F(M)
    r = p

    while t < iterations:
        M = M + encoder.secant_method(M,p)

        with torch.no_grad():
            r_last = r
            r = -encoder.grad_F(M)
            beta = torch.sum( r*r ) / -(torch.sum( p*(r-r_last) ) +1e-6 ) 
            #print('r max: {}, r min: {}, r mean: {}'.format(torch.max(r), torch.min(r),torch.mean(r)))

        p = r + beta * p        
        t += 1
        
        #print('current t: {}, ||r||^2: {}, F: {}'.format(t, torch.sum(r.pow(2)), encoder.F_op(M)))
        print('current t: {}, ||r||^2: {}'.format(t, torch.norm(r)))
        r_record.append(torch.norm(r))
        R_record.append(torch.mean(M[...,2]))
        mse_record.append(imgNRMSE(M,ref_img,brain_mask,TE=0.0115))
        R2 = M[...,2]
        R2[R2<1e-3] = 1e-3
        T2 = 1000 / R2
        nrmse_record.append(NRMSE(T2,varpro_t2,brain_mask,Trange=200))

    #print('M max: {}, M min: {}'.format(torch.max(M[...,:2]), torch.min(M[...,:2])))
    #print('R max: {}, R min: {}, R mean: {}'.format(torch.max(M[...,2]), torch.min(M[...,2]),torch.mean(M[...,2])))
    return M,r_record,R_record,nrmse_record,mse_record
    
itera = 800

torch.manual_seed(3407)
#selected_list = list(range(1,32,4))
selected_list = [1,3,5,7,9,11,13,15]


kspace_data = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/rotated_kdata.pth")[:,:,selected_list,:].to(torch.complex64).cuda()

sensitivity_maps = scio.loadmat(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/coil_sensitivities.mat")['sens']  #(256,208,12)
sensitivity_maps = torch.tensor(sensitivity_maps).to(torch.complex64).cuda()



mask =  torch.from_numpy(scio.loadmat(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/ACC4, masks/ACC4, no ACS.mat")['mask_all']).cuda()    #(256,208,12,8)
#mask = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/ACC6 uniform,26,92-117 are 1.pth").cuda()
#mask = torch.ones(mask.shape).cuda()

mask = mask.permute(0,1,3,2)
print(mask.shape)

print(kspace_data.shape,mask.shape)
kspace_data = kspace_data * mask

mask = mask.contiguous().view((256,208,-1)).permute(2,0,1)
        
M =torch.ones((256,208,3),dtype=torch.float32).cuda()

z = None

reconstructed_M,r_record,R_record,nrmse_record,mse_record = NLCG(kspace_data,sensitivity_maps,mask,M,z=z,iterations=itera)

directory = os.path.join('TE<200result', str(itera)+'4,no acs mask')
if not os.path.exists(directory):
    os.makedirs(directory)

torch.save(reconstructed_M,os.path.join(directory,'recon_result.pth'))
torch.save(r_record,os.path.join(directory,'r.pt')) 
torch.save(R_record,os.path.join(directory,'meanR.pt'))
torch.save(nrmse_record,os.path.join(directory,'nrmse.pt'))
torch.save(mse_record,os.path.join(directory,'mse.pt'))