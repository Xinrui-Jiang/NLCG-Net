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
        self.kimages = kimage_Data  # (256,208,8,12), 8 received image-space images constructed by rss
        self.z = z
        self.mask = mask.permute(0,1,3,2).contiguous().view((256,208,-1)).permute(2,0,1)  # (12*8,256,208)
        self.H, self.W, self.N,self.n_coils = kimage_Data.shape[0], kimage_Data.shape[1], kimage_Data.shape[2],kimage_Data.shape[3]
        self.lam = lam
        self.sens = sens_map  # (256,208,12)
        self.criteria = secant_criteria
        #self.eps = torch.finfo(torch.float32).eps
        self.eps = 1e-6
        if self.N <= 31:
            #self.te_list = [0.023,0.069,0.115,0.161,0.207,0.253,0.299,0.345]
            self.te_list = [0.023,0.046,0.069,0.092,0.115,0.138,0.161,0.184]
            self.te_tensor = torch.ones((self.N * self.n_coils, self.H, self.W)).cuda()
            for i in range(self.N):
                self.te_tensor[i * self.n_coils:(i + 1) * self.n_coils, ...] *= (- self.te_list[i])

    def PFCM_op(self, input_x):
        """
        :param input_x: (256,208,3), (256,208,0) is Mx,(256,208,1) is My, (256,208,2) is v2 = 1/T2
        :return:  PFCM(X),        (12*31,256,208)
        """                    
        M_complex = torch.complex(input_x[..., 0], input_x[..., 1])  # (256,208)

        coil_img = torch.exp(self.te_tensor * input_x[..., 2])
        coil_img = M_complex * coil_img
        coil_img = coil_img * torch.tile(self.sens.permute(2, 0, 1), (self.N, 1, 1))
        coil_img = torch.stack([coil_img.real, coil_img.imag], dim=-1)  # (12*31,256,208,2)

        kspace = fastmri.fft2c(coil_img)
        kspace = torch.complex(kspace[..., 0], kspace[..., 1])  # (12*31,256,208)

        masked_kspace = self.mask * kspace

        return masked_kspace

    def CFH_op(self, x_y):
        """
        :param x_y: (12*8,256,208), PFCM(X) - Y
        :return: CFH(target),      (31,256,208)
        """
        image_space_coil_imgs = fastmri.ifft2c(torch.stack([x_y.real, x_y.imag], dim=-1))
        image_space_coil_imgs = torch.complex(image_space_coil_imgs[..., 0],image_space_coil_imgs[..., 1])  # (12*8,256,208)
        
        image_space_comb = image_space_coil_imgs * torch.tile(torch.conj(self.sens.permute(2, 0, 1)),(self.N, 1, 1))  # (12*8,256,208)
        
        sum_image_space_comb = torch.empty((self.N, self.H, self.W), dtype=torch.complex64).cuda()  # (8,256,208)
        for i in range(self.N):
            sum_image_space_comb[i, :, :] = torch.sum(image_space_comb[i * self.n_coils:(i + 1) * self.n_coils, :, :], dim=0)

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
        """
        :param input_x: (256,208,3), (256,208,0) is Mx,(256,208,1) is My, (256,208,2) is v2 = 1/T2;
        :return: ▽ 0.5 * (|| PFCM(X) - Y || ^2 + lam * || X - Z || ^2), (256,208,3)
        """
        masked_kspace = self.PFCM_op(input_x)

        x_y = masked_kspace - self.kimages.clone().contiguous().view(self.H, self.W, -1).permute(2, 0, 1)  # (12*8,256,208)
        sum_image_space_comb = self.CFH_op(x_y).permute(1, 2, 0)  # (256,208,8)

        gradM = self.grad_M(input_x).permute(0, 1, 3, 2)  # (256,208,3,8*2)
        image_space_comb = torch.stack([sum_image_space_comb.real, sum_image_space_comb.imag], dim=-1)  # (256,208,8,2)
        image_space_comb = image_space_comb.contiguous().view(self.H, self.W, -1)  # (256,208,8*2)

        gradF = torch.matmul(gradM, image_space_comb.unsqueeze(-1)).squeeze()
        
        if self.z != None:
            gradF = gradF + self.lam * (input_x - self.z)
        
        return gradF

    def direction_projector(self,input_x,a,p):
        #to avoid R2 becoming negative, make projection and accordingly adjust direction p
        if a == 0:
            return p
            
        projected_p = p
        step = input_x + a * p
        negative_r2_place = step[...,2] < 1e-4
        
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

                if (torch.abs(a1-a0) < 1e-4) or (count > 10 and error < 1e-3):
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