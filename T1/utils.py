import numpy as np
import torch
import fastmri
import time
import data_consistency as dc

def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):
    #Input_mask: (192,158,32,5), input_data: (192,158,5,32)

    nrow, ncol,echos = input_data.shape[0], input_data.shape[1], input_data.shape[2]

    center_kx = find_center_ind(input_data, axes=(1, 2))
    center_ky = find_center_ind(input_data, axes=(0, 2))
    center_kx,center_ky = center_kx.tolist(),center_ky.tolist()
    for i in range(echos):
        center_kx[i],center_ky[i] = int(center_kx[i]), int(center_ky[i])
    
    temp_mask = input_mask.detach().clone()
    trn_mask,loss_mask = torch.zeros_like(input_mask), torch.zeros_like(input_mask)

    for i in range(echos):
        temp_mask[center_kx[i] - small_acs_block[0] // 2: center_kx[i] + small_acs_block[0] // 2,center_ky[i] - small_acs_block[1] // 2: center_ky[i] + small_acs_block[1] // 2, :, i] = 0
        #图片的主要信号来源处用于训练，不用于计算Loss
        pr = torch.flatten(temp_mask[:,:,0,i]).numpy()
        ind = torch.from_numpy(np.random.choice(np.arange(nrow * ncol),
                                size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr)))
        #在已有的mask中，从没有挡住的部分继续抽，ind是抽取的，抠出来保留信号、用于loss的部分.0.2的比例实际上不完全是按整张图来算的，是按扣掉小方格2x2的大小后计算的
        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))
    
        loss_mask[ind_x, ind_y,:,i] = 1
    
        trn_mask[...,i] = input_mask[...,i] - loss_mask[...,i]

    return trn_mask, loss_mask

def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = torch.sum(torch.square(torch.abs(ref - recon))) / ref.size
    psnr = 20 * torch.log10(torch.abs(ref.max()) / (torch.sqrt(mse) + 1e-10))

    return psnr

def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        #tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
        tensor = torch.norm(tensor,dim=axis,keepdim=keepdims,p=2)

    if not keepdims: return tensor.squeeze()

    return tensor

def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil x echos.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space. Original [1], now [5,]
    """
    echos = kspace.shape[2]
    center_locs = torch.empty((echos,))
    for i in range(echos): 
        density = norm(kspace[...,i,:], axes=axes).squeeze()     #192,158,32
        center_locs[i] = torch.argsort(density)[-1:]

    return center_locs

def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations
    """
    array = torch.zeros(torch.prod(torch.tensor(shape)))
    array[ind] = 1
    ind_nd = torch.nonzero(array.reshape(shape))

    #return [list(ind_nd_ii) for ind_nd_ii in ind_nd]
    return [ind_nd[...,i].tolist() for i in range(ind_nd.shape[-1])]
    
def kspace_transform(M,sens_maps,mask,N_time=5):
    """
    Utilized UnrollNet output, transform it to 31*12 kspace data. mask: 1 192,158,32,5
    sens_maps: (1,192,158,32)
    Returns: (1,192,158,5,32,2)
    """
    batch,H, W, n_coils = sens_maps.shape
    echo_img_list = []
    M_complex = torch.complex(M[..., 0], M[..., 1])  # (1,192,158)
    
    ti_list = [-0.035,-0.15,-0.3,-1.,-3.]
    
    for i in ti_list:
            echo_img = (1. - 2 * torch.exp(i * M[..., 2]))
            echo_img = M_complex * echo_img
            echo_img = echo_img.unsqueeze(1)    #(1,1,192,158)
            echo_img = torch.tile(echo_img,(1,n_coils,1,1))
            echo_img_list.append(echo_img.unsqueeze(1))      #(1,1,32,192,158)

    all_echo_img = torch.cat(echo_img_list,1)                                                                  # (1,5,32,192,158)

    coil_img = all_echo_img * torch.tile(sens_maps.permute(0, 3, 1, 2).unsqueeze(1), (1, N_time, 1, 1, 1)) # (1,5,32,192,158)
    coil_img = torch.stack([coil_img.real, coil_img.imag], dim=-1)  #  (1,5,32,192,158,2)

    kspace = fastmri.fft2c(coil_img)
    kspace = torch.complex(kspace[..., 0], kspace[..., 1])  # (1,5,32,192,158)
    kspace = kspace.permute(0,3,4,1,2)                      # (1,192,158,5,32)
    mask = mask.permute(0,1,2,4,3)                          #1,192,158,5,32
    masked_kspace = kspace * mask                           # (1,192,158,5,32)
    masked_kspace = torch.stack([masked_kspace.real,masked_kspace.imag],dim=-1)
    
    return masked_kspace