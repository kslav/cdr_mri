#!/usr/bin/env python

import torch
import torch.nn
import numpy as np
from copy import copy
from deepinpy.utils import utils

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # define encoding blocks
        self.conv1 = torch.nn.Conv2d(18, 64, 4,2)
        self.batch1 = torch.nn.BatchNorm2d(64)

        self.conv2 = torch.nn.Conv2d(64, 128, 4,2)
        self.batch2 = torch.nn.BatchNorm2d(128)

        self.conv3 = torch.nn.Conv2d(128, 256, 4,2)
        self.batch3 = torch.nn.BatchNorm2d(256)

        self.conv4 = torch.nn.Conv2d(256, 512, 4,2) 
        self.batch4 = torch.nn.BatchNorm2d(512) 

        self.conv5 = torch.nn.Conv2d(512, 512, 4,2) #repeat three times
        self.batch5 = torch.nn.BatchNorm2d(512) 
        self.conv6 = torch.nn.Conv2d(512, 512, 3,2) #repeat three times
        self.conv7 = torch.nn.Conv2d(512, 512, 2,2) #repeat three times


        # used in the encoding block only
        self.leakyrelu  = torch.nn.LeakyReLU(negative_slope=0.2)

        # define decoding blocks
        self.deconv0 = torch.nn.ConvTranspose2d(512, 512, 2,2) 
        self.deconv1 = torch.nn.ConvTranspose2d(1024, 512, 3,2) #repeat three times
        # use batch4 here
        self.deconv2 = torch.nn.ConvTranspose2d(1024, 512, 4,2) 
        self.deconv3 = torch.nn.ConvTranspose2d(1024, 256, 4,2) 
        # use self.batch3 here 
        self.deconv4 = torch.nn.ConvTranspose2d(512, 128, 4,2) 
        # use self.batch2 here
        self.deconv5 = torch.nn.ConvTranspose2d(256, 64, 5,2) 
        self.deconv6 = torch.nn.ConvTranspose2d(128, 3, 4,2) 
        # used in the decoding block only
        self.relu  = torch.nn.ReLU()

    def forward(self, x, T1map_adj):
        ndims = len(x.shape)
        permute_shape = list(range(ndims))
        permute_shape.insert(1, permute_shape.pop(-1))
        x = x.permute(permute_shape)
        temp_shape = x.shape
        x = x.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))
        #print("------> x.shape = ",x.shape)
        _x = x.detach().cpu().numpy()
        np.save("/home/kalina/review/blah.npy", _x)

        # Encoder
        #x is size 224 x 224 x 18
        x1 = self.batch1(self.conv1(x)) #112x112x64
        x2 = self.batch2(self.conv2(self.leakyrelu(x1))) #56x56x128
        x3 = self.batch3(self.conv3(self.leakyrelu(x2))) #28x28x256
        x4 = self.batch4(self.conv4(self.leakyrelu(x3))) #14x14x512
        x5 = self.batch5(self.conv5(self.leakyrelu(x4))) #7x7x512
        x6 = self.batch5(self.conv6(self.leakyrelu(x5))) #2x2x512
        x7 = self.conv7(self.leakyrelu(x6))              #1x1x512 #self.batch5(self.conv7(self.leakyrelu(x6))) #2x2x512
        
        # print("------>x1.shape = ", x1.shape)
        # print("------>x1 max = ", torch.max(x1))
        # print("------>x2.shape = ", x2.shape)
        # print("------>x2 max = ", torch.max(x2))
        # print("------>x3.shape = ", x3.shape)
        # print("------>x3 max = ", torch.max(x3))
        # print("------>x4.shape = ", x4.shape)
        # print("------>x4 max = ", torch.max(x4))
        # print("------>x5.shape = ", x5.shape)
        # print("------>x5 max = ", torch.max(x5))
        # print("------>x6.shape = ", x6.shape)
        # print("------>x6 max = ", torch.max(x6))
        # print("------>x7.shape = ", x7.shape)
        # print("------>x7 max = ", torch.max(x7))


        # Decoder           #input 1x1x512
        y1 = self.deconv0(self.relu(x7)) #2x2x512         #input 2x2x1024
        y2 = self.batch4(self.deconv1(self.relu(torch.cat((y1,x6),dim=1)))) #5x5x512
                                                          #input 5x5x1024
        y3 = self.batch4(self.deconv2(self.relu(torch.cat((y2,x5),dim=1)))) #12x12x512
                                                          #input 12x12x1024
        y4 = self.batch3(self.deconv3(self.relu(torch.cat((y3,x4),dim=1)))) #26x26x256
                                                          #input 26x26x512
        y5 = self.batch2(self.deconv4(self.relu(torch.cat((y4,x3),dim=1)))) #54x54x128
                                                          #input 54x54x256
        y6 = self.batch1(self.deconv5(self.relu(torch.cat((y5,x2),dim=1)))) #112x112x64
                                                          #input 112x112x128
        #print("------->y6.shape=",y6.shape)     
        #print("------->x1.shape=",x1.shape)                          
        y7 =             self.deconv6(self.relu(torch.cat((y6,x1),dim=1))) #224x224x3 (actually 1 x 3 x 224 x 224)

        out = y7*0
        out[:,0,:,:] = y7[:,0,:,:] + T1map_adj.to(x.device)#T1_adj[None,None,...]
        out[:,1,:,:] = y7[:,1,:,:] + torch.mean(x[:,0:8,:,:],1)[:,None,...]
        out[:,2,:,:] = y7[:,2,:,:] + torch.mean(x[:,9:17,:,:],1)[:,None,...]

        #print( "------->y7.shape=",out.shape)

        # print("------->y1.shape=",y1.shape)
        # print("------>y1 max = ", torch.max(y1))
        # print("------->y2.shape=",y2.shape)
        # print("------>y2 max = ", torch.max(y2))
        # print("------->y3.shape=",y3.shape)
        # print("------>y3 max = ", torch.max(y3))
        # print("------->y4.shape=",y4.shape)
        # print("------>y4 max = ", torch.max(y4))
        # print("------->y5.shape=",y5.shape)
        # print("------>y5 max = ", torch.max(y5))
        # print("------->y6.shape=",y6.shape)
        # print("------>y6 max = ", torch.max(y6))
        # print("------->y7.shape=",y7.shape)
        # print("------>y7 max = ", torch.max(y7))
        # print("------>y8 max = ", torch.max(y8))
        return out 

#Purpose: implement the U-Net architecture to perform reconstructions using the RELAX method (Liu et al. 2020 MRM). 
#Tutorial for building the U-Net from scratch: https://amaarora.github.io/2020/09/13/unet.html
#Simple modifications made based on Supporting Figure S2 of Liu et al. 
#Note, this is not a residual U-Net. For T1 mapping, performance of RELAX with a standard and residual U-Net did not significantly differ. 
# class Block defines each convolutional layer of the u-net

# class EncBlock(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(EncBlock, self).__init__()
#         self.leakyrelu  = torch.nn.LeakyReLU(negative_slope=0.2)
#         self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 4,2) #kernel_size=4, stride=2 for all fitlers
#         self.batch = torch.nn.BatchNorm2d(out_ch)
        
#     def forward(self, x):
#         return self.batch(self.conv1(self.leakyrelu(x)))

# class DecBlock(torch.nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DecBlock, self).__init__()
#         self.relu  = torch.nn.ReLU()
#         self.conv2 = torch.nn.ConvTranspose2d(in_ch, out_ch, 4,2) #kernel_size=4, stride=2 for all fitlers
#         self.batch = torch.nn.BatchNorm2d(out_ch)
        
#     def forward(self, x):
#         return self.batch(self.conv2(self.relu(x)))

# # using a combination of blocks, we can build the encoder as a separate class
# class Encoder(torch.nn.Module):
#     def __init__(self, chs=(64, 128, 256, 512, 512, 512, 512)):
#         super(Encoder, self).__init__()
#         #self.enc_blocks = [EncBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
#         self.enc_blocks = torch.nn.ModuleList([EncBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])

#     #it doesn't seem like there's a maxpool operation in the RELAX implementation of the unet, so comment it out
#     def forward(self, x):
#         ftrs = []
#         count = 0
#         for block in self.enc_blocks:
#             count +=1
#             print("-----> count = ",count)
#             x = block(x)
#             print(x.size())
#             ftrs.append(x)
#             #x = self.pool(x)
#         return ftrs

# # combine upsampling blocks to decode the features encoded using the Encoder class
# class Decoder(torch.nn.Module):
#     def __init__(self, chs=(512, 512, 512, 512, 256, 128, 64)):
#         super(Decoder, self).__init__()
#         self.chs         = chs
#         #self.upconvs    = [torch.nn.ConvTranspose2d(chs[i], chs[i+1], 4, 2) for i in range(len(chs)-1)]
#         #self.dec_blocks = [DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]
#         self.upconvs    = torch.nn.ModuleList([torch.nn.ConvTranspose2d(chs[i], chs[i+1], 4, 2) for i in range(len(chs)-1)])
#         self.dec_blocks = torch.nn.ModuleList([DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
#     def forward(self, x, encoder_features):
#         for i in range(len(self.chs)-1):
#             print("----> i = ", i)
#             x        = self.upconvs[i](x)
#             enc_ftrs = encoder_features[i]
#             x        = torch.cat([x, enc_ftrs], dim=1)
#             x        = self.dec_blocks[i](x)
#         return x
    
#     #def crop(self, enc_ftrs, x):
#     #    _, _, H, W = x.shape
#     #    enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
#     #    return enc_ftrs

        

# # The Unet class is defined using the Encoder and Decoder classes
# class UNet(torch.nn.Module):
#     def __init__(self, enc_chs=(64, 128, 256, 512, 512, 512, 512), dec_chs=(512, 512, 512, 512, 256, 128, 64), out_sz=(3,224,224)):
#         super(UNet, self).__init__()
#         self.encoder     = Encoder(enc_chs)
#         self.decoder     = Decoder(dec_chs)


#     def forward(self, x):
#         ndims = len(x.shape)
#         permute_shape = list(range(ndims))
#         permute_shape.insert(1, permute_shape.pop(-1))
#         x = x.permute(permute_shape)
#         temp_shape = x.shape
#         x = x.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))

#         #out_relu  = torch.nn.ReLU()
#         #out_conv2 = torch.nn.ConvTranspose2d(64, 3, 4,2)
#         #in_conv1 = torch.nn.Conv2d(18, 64, 4,2)
#         #in_batch = torch.nn.BatchNorm2d(64)

#         #enc_ftrs = self.encoder(in_batch(in_conv1(x)))
#         #out      = out_conv2(out_relu(self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])))
#         enc_ftrs = self.encoder(x)
#         out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
#         return out



