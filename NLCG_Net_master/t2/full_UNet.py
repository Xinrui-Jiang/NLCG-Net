import torch
import torch.nn as nn

def activation_func(activation, is_inplace=False):
    return nn.ModuleDict([['ReLU', nn.ReLU(inplace=is_inplace)],['LeakyReLU', nn.LeakyReLU(inplace=is_inplace)],
                          ['None', nn.Identity()]])[activation]

def InstanceNorm2d(is_instance_norm, features):
    if is_instance_norm:
        return nn.InstanceNorm2d(features,affine=True)
    else:
        return nn.Identity()
        
class conv_block(nn.Module):
    def __init__(self, filter_size,padding_type,is_instance_norm,activation_type):
        super(conv_block, self).__init__()
        kernel_size, in_c, out_c = filter_size
        self.conv2D = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=padding_type,bias=False)
        self.dropout = nn.Dropout(0.15)
        self.instanceNorm = InstanceNorm2d(is_instance_norm,out_c)
        self.activation_layer = activation_func(activation_type, is_inplace=False)

    def forward(self, x):
        x = self.conv2D(x)
        x = self.instanceNorm(x)
        x = self.activation_layer(x)
        #x = self.dropout(x)
        return x

class double_Convblock(nn.Module):
    def __init__(self, filter_size,padding_type,is_instance_norm,activation_type):
        super(double_Convblock, self).__init__()
        kernel_size,in_c,out_c = filter_size
        self.conv_block1 = conv_block(filter_size,padding_type,is_instance_norm,activation_type)
        self.conv_block2 = conv_block((kernel_size,out_c,out_c),padding_type,is_instance_norm,activation_type)
    
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

# Encoder block: Conv block followed by maxpooling
class encoder_block(nn.Module):
    def __init__(self, filter_size,is_instance_norm,activation_type,padding_type='same'):
        super(encoder_block, self).__init__()
        kernel_size,in_c,out_c = filter_size
        block2_filter_size = (kernel_size,out_c,out_c)
        
        self.conv_block1 = conv_block(filter_size,padding_type,is_instance_norm,activation_type)
        self.conv_block2 = conv_block(block2_filter_size,padding_type,is_instance_norm,activation_type)
        self.maxpooling2d = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        p = self.maxpooling2d(x)
        return x, p


# Decoder block, skip features gets input from encoder for concatenation
class decoder_block(nn.Module):
    def __init__(self, kernel_size,in_c,out_c,is_instance_norm,activation_type,padding_type='same'):
        super(decoder_block, self).__init__()
        conv_outc = int(in_c/2)
        self.conv2DT = nn.ConvTranspose2d(in_c,conv_outc, (2, 2), stride=2)
        self.conv_block1 = conv_block((kernel_size,in_c,out_c),padding_type,is_instance_norm,activation_type)
        self.conv_block2 = conv_block((kernel_size,out_c,out_c),padding_type,is_instance_norm,activation_type)

    def forward(self, x, skip_features):
        x = self.conv2DT(x)
        x = torch.cat((x, skip_features),dim=1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x
         
class U_net(nn.Module):
    def __init__(self,in_ch=3):
        super(U_net, self).__init__()
        kernel_size = 3
        filter1 = (kernel_size,in_ch,32)
        filter2 = (kernel_size,32,64)
        filter3 = (kernel_size,64,128)
        filter4 = (kernel_size,128,256)
        
        self.encoder_block1 = encoder_block(filter1,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.encoder_block2 = encoder_block(filter2,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.encoder_block3 = encoder_block(filter3,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        
        #self.double_Convblock = double_Convblock(filter4,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.double_Convblock = encoder_block(filter4,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        
        self.decoder_block1 = decoder_block(kernel_size,256,128,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.decoder_block2 = decoder_block(kernel_size,128,64,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.decoder_block3 = decoder_block(kernel_size,64,32,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.final_layer = nn.Conv2d(in_channels=32, out_channels=in_ch, kernel_size=1, padding='same',bias=False)
        
        #self.double_conv = double_Convblock(filter5,is_instance_norm=True,activation_type='ReLU',padding_type='same')
        #self.final_conv = conv_block(filter6,is_instance_norm=False,activation_type='None',padding_type='same')
        
        self.M_activate_layer = nn.Tanh()
        self.R_activate_layer = nn.Sigmoid()    
        
        #self.scaling_factor = torch.ones((1,3,1,1),dtype=torch.float32).cuda()
        #self.scaling_factor[:,2] = self.scaling_factor[:,2] * 40
        scaling_factor = torch.ones((1,3,1,1),dtype=torch.float32)
        scaling_factor[:,2] = scaling_factor[:,2] * 40.
        self.scaling_factor = nn.Parameter(scaling_factor)
        print(self.scaling_factor.is_leaf,self.scaling_factor.requires_grad)
      
    def forward(self, x):
        s1, p1 = self.encoder_block1(x)
        s2, p2 = self.encoder_block2(p1)
        s3, p3 = self.encoder_block3(p2)
        
        p4 = self.double_Convblock(p3)

        out = self.decoder_block1(p4,s3)
        out = self.decoder_block2(out,s2)
        out = self.decoder_block3(out,s1)
        out = self.final_layer(out)
        
        """
        #scaling and rescaling type
        out = out * minmax_list[1]
        out = out + minmax_list[0]
        final_r_out = self.relu(out[:,2:,...])
        """
        #print(torch.max(out[:,:2,...]),torch.min(out[:,:2,...]),torch.mean(out[:,:2,...]),torch.std(out[:,:2,...]))
        #print('OUTPUT no activate M')
        #print(torch.max(out[:,2,...]),torch.min(out[:,2,...]),torch.mean(out[:,2,...]),torch.std(out[:,2,...]))
        #print('OUTPUT no activate R')
        
        m_out = self.M_activate_layer(out[:,0:2,...])
        r_out = self.R_activate_layer(out[:,2:,...]) 

        final_out = torch.cat([m_out,r_out],dim=1)
        final_out = final_out * self.scaling_factor
        #print(torch.max(m_out),torch.min(m_out),torch.mean(m_out),torch.std(m_out))
        #print('OUTPUT activate M')
        #print(torch.max(r_out),torch.min(r_out),torch.mean(r_out),torch.std(r_out))
        #print('OUTPUT activate R')
        

        return final_out