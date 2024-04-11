import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='NLCG-Net: Nonlinear Conjugate Gradient Network for quantitative MRI')

    # %% hyperparameters for the data path
    parser.add_argument('--data_opt', type=str, default="T2",
                    help='type of dataset')
    parser.add_argument('--data_dir', type=str, default=r"./dataset/T2",
                    help='data directory')   
    parser.add_argument('--SSLdata_dir', type=str, default=r"./dataset/T2/k_0.2428_r_10_acc_8",
                    help='data directory for predivided {num_reps} mask sets')
    
    parser.add_argument('--kdata_name', type=str, default="rotated_8kdata.pth",
                    help='name of k-space data')
    parser.add_argument('--sens_name', type=str, default="coil_sensitivities.mat",
                    help='name of coil sensitivity data')
    parser.add_argument('--initial_guess_name', type=str, default="k_0.2428_r_10_ACC_8_init_guess.pth",
                    help='name of data for NLCG initial guess')
    
    parser.add_argument('--ref_data_dir', type=str, default=r"./reference_dataset/T2",
                    help='ground truth reference directory')                    
    parser.add_argument('--ref_r_name', type=str, default="ground_truth.pth",
                    help='name of reference R2/R1 data')
    parser.add_argument('--ref_img_name', type=str, default=r"reconstructed_8img.pth",
                    help='name of reference image space data')
    parser.add_argument('--csf_mask_name', type=str, default="csf_block_mask.pth",
                    help='name of CSF mask data for calculating NRMSE/imgNRMSE')

                    
    # %% hyperparameters for the data                         
    parser.add_argument('--acc_rate', type=int, default=8,
                        help='acceleration rate')
    parser.add_argument('--nrow_GLOB', type=int, default=256,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=208,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--necho_GLOB', type=int, default=8,
                        help='number of echos in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=12,
                        help='number of coils of the slices in the dataset') 
    parser.add_argument('--echo_time_list', type=list, default=[0.23,0.46,0.69,0.92,1.15,1.38,1.61,1.84],
                        help='default echo time with "ms" as unit for echo images')   
    
    # %% hyperparameters for scaling
    parser.add_argument('--k_scaling_factor', type=float, default=0.2428,
                    help='scaling factor for M0')
    parser.add_argument('--r_scaling_factor', type=float, default=10.,
                    help='scaling factor for R_telda = R / r_scaling_factor')
    parser.add_argument('--secant_criteria', type=float, default=1e-6,
                    help='secant criteria in deciding convergence over NLCG')
    
    # %% hyperparameters for the network
    parser.add_argument('--model_type', type=str, default='ResNet',
                        help='Type of the used model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device for training')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=6,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=15,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=10,
                        help='number of Nonlinear Conjugate Gradient iterations for DC')

    # %% hyperparameters for the NLCG-Net training
    parser.add_argument('--rho_val', type=float, default=0.2,
                        help='cardinality of the validation mask')                        
    parser.add_argument('--rho_train', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--num_reps', type=int, default=25,
                        help='number of repetions for the remainder mask')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')
    parser.add_argument('--TL_path', type=str, default=None,
                        help='path to pretrained model')                                        
    parser.add_argument('--stop_training', type=int, default=25,
                        help='stop training if a new lowest validation loss hasnt been achieved in xx epochs')                    

    parser.add_argument('--saved_model_name', type=str, default='ZS_SSL_Model_300Epochs_Rate4_10Unrolls',
                        help='model name to be used for eval')       
    return parser
