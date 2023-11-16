import torch
import numpy as np
import math
import fastmri
import time
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio

secant_criteria = 1e-6

"""
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

def imgNRMSE(result,ref_img,mask,TI_tensor):
    #calculate mean imgNRMSE over 5 images    
    #ref_img: (192,158,5)
    #mask: (192,158)
    #TI_tensor: (192,158,5)
    #result: (192,158,3)

    mask = mask.unsqueeze(-1)
    num = img.shape[-1]
    M = torch.abs(torch.complex(result[...,0],result[...,1]))
    img = (1 - 2 * torch.exp(TI_tensor*result[...,-1])) * M

    imgnrmse = torch.norm((img-ref_img)*mask,dim=(0,1)) / torch.norm(ref_img*mask,dim=(0,1))
    avg_nrmse = torch.sum(imgnrmse) / num
    return avg_nrmse  
"""

class operator():
    def __init__(self, kimage_Data, sens_map, mask, z, lam=0.001):
        self.kimages = kimage_Data  # (192,158,5,32), 5 received image-space images
        self.z = z
        self.mask = mask            # (5 * 32, 192,158), coil first
        self.H, self.W, self.N,self.n_coils = kimage_Data.shape[0], kimage_Data.shape[1], kimage_Data.shape[2],kimage_Data.shape[3]
        self.lam = lam
        self.sens = sens_map  # (192,158,32)
        self.criteria = secant_criteria
        self.eps = 1e-6
        self.ti_list = [0.035,0.15,0.3,1.,3.]
        #self.ti_list = [0.07,0.3,0.6,2.,6.]
        #self.ti_list = [0.14,0.6,1.2,4.,12.]
        self.ti_tensor = torch.ones((self.N * self.n_coils, self.H, self.W)).cuda()
        for i in range(self.N):
            self.ti_tensor[i * self.n_coils:(i + 1) * self.n_coils, ...] *= (- self.ti_list[i])

    def PFCM_op(self, input_x):
        """
        :param input_x: (192,158,3), (192,158,0) is Mx,(192,158,1) is My, (192,158,2) is R1 = 1/T1
        :return:  PFCM(X),        (5*32,192,158)
        """                    
        M_complex = torch.complex(input_x[..., 0], input_x[..., 1])  # (192,158)

        coil_img = torch.exp(self.ti_tensor * input_x[..., 2])
        coil_img = M_complex * (1- 2* coil_img)                                               #first modification
        coil_img = coil_img * torch.tile(self.sens.permute(2, 0, 1), (self.N, 1, 1))              #follows coil first
        coil_img = torch.stack([coil_img.real, coil_img.imag], dim=-1)  # (5*32,192,158,2)

        kspace = fastmri.fft2c(coil_img)
        kspace = torch.complex(kspace[..., 0], kspace[..., 1])  # (5*32,256,208)

        masked_kspace = self.mask * kspace

        return masked_kspace

    def CFH_op(self, x_y):
        """
        :param x_y: (5*32,192,158), PFCM(X) - Y
        :return: CFH(target)      (5,192,158)
        """
        image_space_coil_imgs = fastmri.ifft2c(torch.stack([x_y.real, x_y.imag], dim=-1))
        image_space_coil_imgs = torch.complex(image_space_coil_imgs[..., 0],image_space_coil_imgs[..., 1])  # (5*32,192,158)
        
        image_space_comb = image_space_coil_imgs * torch.tile(torch.conj(self.sens.permute(2, 0, 1)),(self.N, 1, 1))  # (5*32,192,158)
        
        sum_image_space_comb = torch.empty((self.N, self.H, self.W), dtype=torch.complex64).cuda()  # (5,192,158)
        for i in range(self.N):
            sum_image_space_comb[i, :, :] = torch.sum(image_space_comb[i * self.n_coils:(i + 1) * self.n_coils, :, :], dim=0)

        return sum_image_space_comb

    def grad_M(self, input_x):
        """
        :param input_x: (192,158,3), (192,158,0) is Mx,(192,158,1) is My, (192,158,2) is R1 = 1/T1
        :return: (e^(-t*v2), -tM0e^(-t*v2), (256,208,5*2,3)         #first within same echo, then separate real/complex first
        """
        gradM = torch.zeros((self.H, self.W, self.N, 2, 3), dtype=torch.float32).cuda()

        for i,j in enumerate(self.ti_list):
            gradM[:, :, i, 0, 0] = 1 - 2 * torch.exp(-j * input_x[:, :, 2])  # gradient of M(X) to M0
            gradM[:, :, i, 1, 1] = 1 - 2 * torch.exp(-j * input_x[:, :, 2])
            gradM[:, :, i, 0, 2] = 2 * j * input_x[:, :, 0] * torch.exp(-j * input_x[:, :, 2])
            gradM[:, :, i, 1, 2] = 2 * j * input_x[:, :, 1] * torch.exp(-j * input_x[:, :, 2])

        gradM = gradM.contiguous().view(self.H, self.W, -1, 3)
        #gradM = gradM * 2                                                      #2ND Modification
        
        print('M gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradM[...,:2]),torch.min(gradM[...,:2]),torch.mean(torch.abs(gradM[...,:2]))))
        print('R gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradM[...,2]),torch.min(gradM[...,2]),torch.mean(torch.abs(gradM[...,2]))))
        #time.sleep(1e-3)

        return gradM

    def grad_F(self, input_x):

        masked_kspace = self.PFCM_op(input_x)      #(5*32,192,158),same echo first

        x_y = masked_kspace - self.kimages.clone().contiguous().view(self.H, self.W, -1).permute(2, 0, 1)  # (5*32,192,158)
        sum_image_space_comb = self.CFH_op(x_y).permute(1, 2, 0)  # (192,158,5)

        gradM = self.grad_M(input_x).permute(0, 1, 3, 2)  # (192,158,3,5*2)
        image_space_comb = torch.stack([sum_image_space_comb.real, sum_image_space_comb.imag], dim=-1)  # (192,158,5,2)
        image_space_comb = image_space_comb.contiguous().view(self.H, self.W, -1)  # (192,158,5*2)

        gradF = torch.matmul(gradM, image_space_comb.unsqueeze(-1)).squeeze()  # (192,158,3)
        
        if self.z != None:
            gradF = gradF + self.lam * (input_x - self.z)
        
        print('F M gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradF[...,:2]),torch.min(gradF[...,:2]),torch.mean(torch.abs(gradF[...,:2]))))
        print('F R gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradF[...,2]),torch.min(gradF[...,2]),torch.mean(torch.abs(gradF[...,2]))))
        #time.sleep(1e-3)
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
        dFdx = self.grad_F(new_step)            #(192,158,3)
        dFda = torch.sum(dFdx * p)  # (192,158,1,1)
        
        return dFda   
            
    def secant_method(self, input_x, p):
        """
        apply Line Search to find the curent best step size alpha, which suffice:
        argmin F(x + alpha * p), i.e., dF/d alpha = 0. 
        as long as dF/d alpha is close to a small number, we can claim that we have got a correct alpha
        :param input_x: (192,158,3), (192,158,0) is Mx,(192,158,1) is My, (192,158,2) is R1 = 1/T1
        :param p: current step direction, (192,158,3), (192,158,0) is Mx direction, (192,158,2) is T1 direction
        :param z: regularization term z, (192,158,3)
        :return: alpha, current best step size, (192,158,3)
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
            print('count: {}, error: {}, ddy: {}, d1: {}, d0: {}, a1: {},a0: {}'.format(count, error,ddy,d1,d0,a1,a0))

            a0 = a1
            a1 = a_new
            count += 1

            if (torch.abs(a1-a0) < 1e-5) or (count > 10 and error < 1e-5):
                break
            if count > 20:
                break
                
        print('choose a: {}'.format(a1))
        print('choose a: {}, step max: {}, step min: {}, step mean: {}'.format(a1,torch.max(a1 * p_normalize), torch.min(a1 * p_normalize),torch.mean(a1 * p_normalize)))
        best_step = a_new * p_new

        return best_step

def NLCG(kimage_Data, sens_map, mask, M, z=None, iterations=300, lam=0.001, criteria=1e-6,TE=0.0115):
    """
    #perform non-linear conjugate gradient method to reconstruct (M,T2) from 5 images constructed by rss
    :param image_Data: 31 images reconstructed by RSS, (192,158,5)
    :param sens_map: 12 sensitivity maps, (192,158,32)
    :param mask: Accelerate map, (192,158)
    :param M: Corresponding M,R2 to be estimated, can be all one(initialize) or from last NL-CG block output,(192,158,3)
    :param z: Last regularizer output
    :param iterations: NL-CG iteration number
    :param lam: Regularizartion parameter
    :param criteria: Stopping criteria used in secant methods
    :param condition: Choose RF or Daiyuan method in NL-CG
    :param TE: Interleave time, using 's' as unit
    :projection: Decide if performing projection to let all R1_hat be non-negative
    :return: Reconstructed (M,T2), (192,158,3)
    """
    varpro_t2 = torch.from_numpy(scio.loadmat(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/2to32,8data/0-1000 2to32 8 point T2.mat")['T2std']).cuda()
    brain_mask = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/2to32,8data/csf_block_mask.pth").cuda()
    selected_list = list(range(1,32,4))
    selected_list = [1,3,5,7,9,11,13,15]
    ref_img = torch.load(r"/root/autodl-tmp/MOBA-T2mapping-master/dataset/reconstructed_img.pth")[:,:,selected_list].cuda()

    r_record,R_record,nrmse_record,mse_record = [],[],[],[]
    
    encoder = operator(kimage_Data=kimage_Data, sens_map=sens_map, mask=mask, z=z, lam=lam)
    
    t = 0
    p = -encoder.grad_F(M)
    r = p

    while t < iterations:
        M = M + encoder.secant_method(M,p)

        with torch.no_grad():
            r_last = r
            r = -encoder.grad_F(M)
            beta = torch.sum( r*r ) / -(torch.sum( p*(r-r_last) ) +1e-6 ) 
            print('r max: {}, r min: {}, r mean: {}'.format(torch.max(r), torch.min(r),torch.mean(r)))

        p = r + beta * p        
        t += 1
        
        #print('current t: {}, ||r||^2: {}, F: {}'.format(t, torch.sum(r.pow(2)), encoder.F_op(M)))
        print('current t: {}, ||r||^2: {}'.format(t, torch.norm(r)))
        r_record.append(torch.norm(r))
        #R_record.append(torch.mean(M[...,2]))
        #mse_record.append(imgNRMSE(M,ref_img,brain_mask,TE=0.0115))
        #nrmse_record.append(NRMSE(T2,varpro_t2,brain_mask,Trange=200))

    #print('M max: {}, M min: {}'.format(torch.max(M[...,:2]), torch.min(M[...,:2])))
    #print('R max: {}, R min: {}, R mean: {}'.format(torch.max(M[...,2]), torch.min(M[...,2]),torch.mean(M[...,2])))
    return M,r_record,R_record,nrmse_record,mse_record
    
itera = 800
ACC = 6
if_ACS = False
scale_rate = 0
torch.manual_seed(3407)

data_dirt = r"/root/autodl-tmp/MOBA-T2mapping-master/T1 mapping dataset/"

sensitivity_maps = torch.load(os.path.join(data_dirt,"sens_map_11.pth")).cuda()                #192, 158, 32
mask = torch.load(os.path.join(data_dirt,"ACC6,noACS.pth")).permute(0,1,3,2).cuda()            #192,158,5,32
kspace_data = torch.load(os.path.join(data_dirt,"5TIimg_11.pth")).permute(0,1,3,2).cuda()      #192,158,5,32




kspace_data = kspace_data * mask

print('max value: {}'.format(torch.max(torch.abs(kspace_data))))

kspace_data = kspace_data / torch.max(torch.abs(kspace_data))

time.sleep(3)

mask = mask.contiguous().view((192,158,-1)).permute(2,0,1)                                      #5*32,192,158, firstly stack coil, then stack different TI
        
M =torch.ones((192,158,3),dtype=torch.float32).cuda()

z = None


reconstructed_M,r_record,R_record,nrmse_record,mse_record = NLCG(kspace_data,sensitivity_maps,mask,M,z=z,iterations=itera)

directory = os.path.join(r"/root/autodl-tmp/MOBA-T2mapping-master/T1Result/noscale", str(itera)+'iter'+str(ACC)+'ACC'+str(int(if_ACS))+'is_ACS'+str(scale_rate)+'scale')
if not os.path.exists(directory):
    os.makedirs(directory)

torch.save(reconstructed_M,os.path.join(directory,'recon_result.pth'))
torch.save(r_record,os.path.join(directory,'r.pt')) 
#torch.save(R_record,os.path.join(directory,'meanR.pt'))
#torch.save(nrmse_record,os.path.join(directory,'nrmse.pt'))
#torch.save(mse_record,os.path.join(directory,'mse.pt'))