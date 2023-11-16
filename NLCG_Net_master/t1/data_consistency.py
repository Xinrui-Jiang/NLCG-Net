import torch
import numpy as np
import math
import fastmri
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
"""
#1.删了recordingM 2. 删了inputx.tofloat32 3.改了secant的输出内容
"""
secant_criteria = 1e-6

class operator():
    def __init__(self, kimage_Data, sens_map, mask, z, lam=0.001):
        self.kimages = kimage_Data  # (192,158,5,32), 5 received image-space images
        self.z = z
        self.H, self.W, self.N,self.n_coils = kimage_Data.shape[0], kimage_Data.shape[1], kimage_Data.shape[2],kimage_Data.shape[3]
        #print('check mask shape, should be 192,158,32,5')
        #print(mask.shape)
        self.mask = mask.permute(0,1,3,2).contiguous().view((self.H,self.W,-1)).permute(2,0,1)            # (5 * 32, 192,158), coil first
        self.lam = lam
        self.sens = sens_map  # (192,158,32)
        self.criteria = secant_criteria
        self.eps = 1e-6
        self.ti_list = [0.035,0.15,0.3,1.,3.]
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
        coil_img = M_complex * (1- 2 * coil_img)                                               #first modification
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
        
        #print('M gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradM[...,:2]),torch.min(gradM[...,:2]),torch.mean(torch.abs(gradM[...,:2]))))
        #print('R gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradM[...,2]),torch.min(gradM[...,2]),torch.mean(torch.abs(gradM[...,2]))))

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
        
        #print('F M gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradF[...,:2]),torch.min(gradF[...,:2]),torch.mean(torch.abs(gradF[...,:2]))))
        #print('F R gradient: max: {}, min: {}, abs mean: {}'.format(torch.max(gradF[...,2]),torch.min(gradF[...,2]),torch.mean(torch.abs(gradF[...,2]))))
        return gradF

    def direction_projector(self,input_x,a,p):
        #to avoid R2 becoming negative, make projection and accordingly adjust direction p
        if a == 0:
            return p
            
        projected_p = p
        step = input_x + a * p
        negative_r2_place = step[...,2] < 0
        
        if torch.any(negative_r2_place) == True:
            projected_p[...,2][negative_r2_place] = -input_x[...,2][negative_r2_place] / (a+self.eps)
         
        return projected_p
    
    def dFd_alpha(self,input_x,alpha,p):        # calculate dF/d alpha, following chain rule

        new_step = input_x + alpha * p
        dFdx = self.grad_F(new_step)            #(256,208,3)
        dFda = torch.sum(dFdx * p)
        
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

        #p_normalize = p / torch.sqrt(torch.sum(p ** 2))

        error = self.criteria + 1
        count = 0

        while (error > self.criteria) and (count<20):
            with torch.no_grad():
                p1 = self.direction_projector(input_x,a1,p)
                p0 = self.direction_projector(input_x,a0,p)
                
                #print('examine p1 = p0: {}'.format(torch.all(p1==p0)))
                
                d1 = self.dFd_alpha(input_x, a1, p1)
                d0 = self.dFd_alpha(input_x, a0, p0)
                ddy = (d1 - d0) / ( (a1 - a0)+self.eps )      
                a_new = a1 - d1 / (ddy+self.eps)
            
                p_new = self.direction_projector(input_x,a_new,p)
                error = torch.abs(self.dFd_alpha(input_x, a_new, p_new))
                #print('count: {}, error: {}, ddy: {}, d1: {}, d0: {}, a1: {},a0: {}'.format(count, error,ddy,d1,d0,a1,a0))

                a0 = a1
                a1 = a_new
                count += 1

            if (torch.abs(a1-a0) < 1e-5) or (count > 10 and error < 1e-5):
                    break
                
        #print('choose a: {}'.format(a1))
        #print('choose a: {}, step max: {}, step min: {}, step mean: {}'.format(a1,torch.max(a1 * p_normalize), torch.min(a1 * p_normalize),torch.mean(a1 * p_normalize)))
        best_step = a_new * self.direction_projector(input_x,a_new,p)
        
        return best_step

def NLCG(kimage_Data, sens_map, mask, M, lam, z=None, iterations=300):
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
    encoder = operator(kimage_Data, sens_map, mask, z, lam)

    t = 0
    p = -encoder.grad_F(M)
    r = p

    while t < iterations:
        M = M + encoder.secant_method(M,p)

        #with torch.no_grad():
        r_last = r
        r = -encoder.grad_F(M)
        beta = torch.sum( r*r ) / -(torch.sum( p*(r-r_last) ) +1e-6 ) 
        p = r + beta * p        
        t += 1
        
        #print('current t: {}, ||r||^2: {}'.format(t, torch.norm(r)))

    print('M max: {}, M min: {}'.format(torch.max(M[...,:2]), torch.min(M[...,:2])))
    print('R max: {}, R min: {}, R mean: {}'.format(torch.max(M[...,2]), torch.min(M[...,2]),torch.mean(M[...,2])))
    return M

def dc_block(kimages,sens_maps,mask,M,lam, z=None,iterations=300,batchSize=1):
    """
    DC block employs non-linear conjugate gradient for data consistency
    """
    cg_recons = []
    for ii in range(batchSize):
        cg_recon = NLCG(kimages[ii], sens_maps[ii],mask[ii],M[ii],lam,z[ii],iterations)
        cg_recons.append(cg_recon.unsqueeze(0))
    dc_block_recons = torch.cat(cg_recons, 0)

    return dc_block_recons