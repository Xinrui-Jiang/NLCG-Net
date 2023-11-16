import torch
import torch.nn as nn
def activation_func(activation, is_inplace=False):
    return nn.ModuleDict([['LeakyReLU', nn.ReLU(inplace=is_inplace)],
                          ['None', nn.Identity()]])[activation]

def InstanceNorm(is_instance_norm, features):
    if is_instance_norm:
        return nn.InstanceNorm2d(features)
    else:
        return nn.Identity()
"""      
def LayerNorm(is_layer_norm, features):
    if is_layer_norm:
        return nn.LayerNorm(features)
    else:
        return nn.Identity()
"""
def scaling_layer(is_scaling=True):
    scale_factor = torch.tensor([0.1], dtype=torch.float32)
    if is_scaling:
        return nn.Mul(scale_factor)
    else:
        return nn.Identity()


def conv_layer(filter_size, padding=1, is_instance_norm=True, activation_type='LeakyReLU', *args, **kwargs):
    kernel_size, in_c, out_c = filter_size
    return nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=padding, bias=False),
        InstanceNorm(is_instance_norm, (out_c,256,208)),
        activation_func(activation_type))


def ResNetBlock(filter_size):
    return nn.Sequential(conv_layer(filter_size, activation_type='LeakyReLU'),
                         conv_layer(filter_size, activation_type='None'))


class ResNetBlocksModule(nn.Module):
    def __init__(self, device, filter_size, num_blocks):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([ResNetBlock(filter_size=filter_size) for _ in range(num_blocks)])
        self.trace = []

    def forward(self, x):
        scale_factor = torch.tensor([0.1], dtype=torch.float32).to(self.device)
        for layer in self.layers:
            x = x + layer(x) * scale_factor

        return x


class ResNet(nn.Module):
    def __init__(self, device,in_ch=3, num_of_resblocks=5):
        super(ResNet, self).__init__()
        self.in_ch = in_ch
        self.num_of_resblocks = num_of_resblocks
        self.device = device

        kernel_size = 3
        out_kernel_size = 1
        padding = 1
        features = 64
        filter1 = [kernel_size, in_ch, features]  # map input to size of feature maps
        filter2 = [kernel_size, features, features]  # ResNet Blocks
        filter3 = [out_kernel_size, features, in_ch]  # map output channels  to input channels

        self.layer1 = conv_layer(filter_size=filter1,is_instance_norm=True,activation_type='LeakyReLU')  
        self.layer2 = ResNetBlocksModule(device=self.device, filter_size=filter2, num_blocks=num_of_resblocks)
        #self.layer3 = conv_layer(filter_size=filter2, activation_type='None')
        self.layer3 = conv_layer(filter_size=filter3, padding=0,is_instance_norm=False,activation_type='None')

    def forward(self, input_x):
        l1_out = self.layer1(input_x)        
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        #l4_out = self.layer4(l3_out)
        #temp = l3_out + l1_out
        #nwl4_out = self.layer4(temp)
        nw_out = l3_out + input_x
        return nw_out
        
def activation_func(activation, is_inplace=False):
    return nn.ModuleDict([['ReLU', nn.ReLU(inplace=is_inplace)],['LeakyReLU', nn.LeakyReLU(inplace=is_inplace)],
                          ['None', nn.Identity()]])[activation]

def InstanceNorm2d(is_instance_norm, features):
    if is_instance_norm:
        return nn.InstanceNorm2d(features,affine=False)
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
        
        self.double_Convblock = double_Convblock(filter4,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        
        self.decoder_block1 = decoder_block(kernel_size,256,128,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.decoder_block2 = decoder_block(kernel_size,128,64,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.decoder_block3 = decoder_block(kernel_size,64,32,is_instance_norm=True,activation_type='LeakyReLU',padding_type='same')
        self.final_layer = nn.Conv2d(in_channels=32, out_channels=in_ch, kernel_size=1, padding='same',bias=False)
        
        #self.double_conv = double_Convblock(filter5,is_instance_norm=True,activation_type='ReLU',padding_type='same')
        #self.final_conv = conv_block(filter6,is_instance_norm=False,activation_type='None',padding_type='same')
        
        self.M_activate_layer = nn.Tanh()
        self.R_activate_layer = nn.Sigmoid()    
        
        self.scaling_factor = torch.ones((1,3,1,1),dtype=torch.float32)
        self.scaling_factor[:,2] = self.scaling_factor[:,2] * 50
      
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
def activation_func(activation, is_inplace=False):
    return nn.ModuleDict([['ReLU', nn.ReLU(inplace=is_inplace)],['LeakyReLU', nn.LeakyReLU(inplace=is_inplace)],
                          ['None', nn.Identity()]])[activation]

def InstanceNorm2d(is_instance_norm, features):
    if is_instance_norm:
        return nn.InstanceNorm2d(features,affine=False)
    else:
        return nn.Identity()
        
class conv_unit(nn.Module):
    def __init__(self, filter_size,padding_type,is_instance_norm,activation_type):
        super(conv_unit, self).__init__()
        kernel_size, in_c, out_c = filter_size
        self.conv2D = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=padding_type,bias=False)
        self.instanceNorm = InstanceNorm2d(is_instance_norm,out_c)
        self.activation_layer = activation_func(activation_type, is_inplace=False)

    def forward(self, x):
        x = self.conv2D(x)
        x = self.instanceNorm(x)
        x = self.activation_layer(x)

        return x
        
class ConvBlocks(nn.Module):
    def __init__(self,filter_size,padding_type,is_instance_norm,activation_type,num_blocks):
        super().__init__()
        self.layers = nn.ModuleList([conv_unit(filter_size,padding_type,is_instance_norm,activation_type) for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.layers:  
            x = layer(x)
            
        return x

max_val_start = torch.ones((1,3,1,1),dtype=torch.float32)
max_val_start[:,2] = max_val_start[:,2] * 38
#max_val_start[:,:2] = max_val_start[:,:2] * 0.2
         
class SimpleNet(nn.Module):
    def __init__(self,device,in_ch=3,num_blocks=14):
        super(SimpleNet, self).__init__()
        kernel_size = 3
        filter0 = (kernel_size,in_ch,64)
        filter1 = (kernel_size,64,64)
        filter2 = (1,64,in_ch)
  
        self.conv0 = conv_unit(filter_size=filter0,padding_type='same',is_instance_norm=True,activation_type='LeakyReLU')
        self.conv1 = ConvBlocks(filter_size=filter1,padding_type='same',is_instance_norm=True,activation_type='LeakyReLU',num_blocks=num_blocks)
        self.conv2 = conv_unit(filter_size=filter2,padding_type='same',is_instance_norm=True,activation_type='None')
        self.max_val = nn.Parameter(max_val_start)
        
        self.M_activate_layer = nn.Tanh()
        self.R_activate_layer = nn.Sigmoid()    
    
    
    def forward(self, x):
        feature = self.conv0(x)
        feature = self.conv1(feature)
        feature = self.conv2(feature)
        
        m_out = self.M_activate_layer(feature[:,0:2,...])
        r_out = self.R_activate_layer(feature[:,2:,...]) 
        out = torch.cat([m_out,r_out],dim=1)
        
        out = out * self.max_val

        return out
        
        
if __name__ == '__main__':
    net = U_net(3)
    #net = SimpleNet(device='cpu',in_ch=3,num_blocks=14)
    #net = ResNet( device='cpu',in_ch=3, num_of_resblocks=5)
    #-------------1--------------
    sum_ = 0
    for name, param in net.named_parameters():
        mul = 1
        for size_ in param.shape:
    	      mul *= size_							
        sum_ += mul									
        print('%14s : %s' % (name, param.shape))  
        # print('%s' % param)						
    print('param num:', sum_)					
    """
    # -------------2--------------
    for param in net.parameters():
        print(param.shape)
        #print(param)
    
    # -------------3--------------
    params = list(net.parameters())
    for param in params:
        print(param.shape)
        # print(param)
    """