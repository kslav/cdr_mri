#!/usr/bin/env python

# RELAX is from Liu et al. 2020. Mag Resn Med (DOI: 10.1002/mrm.28659). 
# While the original implementation used real-valued, simulated images and trained over many batches, 
# we modified the U-net model to take in complex-valued images as input and to train iteratively over
# a single batch in an untrained fashion. 

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import UNet
from deepinpy.recons import Recon
from deepinpy.utils import utils
import torch
import numpy as np

class RelaxRecon(Recon):

    def __init__(self, args):
        super(RelaxRecon, self).__init__(args)
        self.network = UNet()
        self.FAs = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.TR = 6 #ms
        self.sina = [ 0.06975641,  0.10452838,  0.13917298,  0.17364803,  0.20791152, 0.2419217 ,  0.27563713,  0.30901674,  0.34201987]
        self.cosa = [ 0.99756405,  0.9945219 ,  0.99026809,  0.98480778,  0.97814764, 0.97029578,  0.96126176,  0.9510566 ,  0.93969272]
        self.T1map = torch.from_numpy(np.load("/home/kalina/review/t1map_Sub8_gt.npy")) #T1 map of adjoint pre-computed and loaded

    def batch(self, data):

        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)
        self.x_adj = self.A.adjoint(inp)

        np.save("/home/kalina/review/Sub8_R12_x_adj.npy",utils.t2n(self.x_adj[0,...]) )

    def signal_model(self, T1map, PD_real, PD_imag):
        im = self.x_adj*0
        T1map = self.T1map.to(self.x_adj.device)#T1map*5000
        #T1map = T1map*np.max(self.T1map.detach().cpu().numpy())
        #PD_real = np.sqrt(np.linalg.norm(self.x_adj.detach().cpu().numpy()))*PD_real # or multiply by whatever sqrt(x_adj norm is)
        #PD_imag = np.sqrt(np.linalg.norm(self.x_adj.detach().cpu().numpy()))*PD_imag

        PD_real = np.sqrt(np.max(self.x_adj[...,0].detach().cpu().numpy()))*PD_real # or multiply by whatever sqrt(x_adj norm is)
        PD_imag = np.sqrt(np.max(self.x_adj[...,1].detach().cpu().numpy()))*PD_imag 

        T1map[T1map>5000]=5000
        T1map[T1map<1] = 1
                            # 224 x 224
        E1 = torch.exp(-self.TR/T1map)
        #SPGRE equation

        for i in range(0, np.size(self.FAs)):
            #sina = np.sin(3.14159 / 180 *self.FAs[i]) # make this faster in torch
            #cosa = np.cos(3.14159/ 180 *self.FAs[i])
            z = self.sina[i] * (1 - E1) / (1 - self.cosa[i] * E1)
            z_final = z
            #np.save("/home/kalina/review/z_blah.npy",z)
            #print("z_final.shape = ",z_final.shape)
            #im_z[:,i,:,:,0] = z # store z so that you can compute the norm of im_z

            im[:,i,:,:,0] = z_final*PD_real 
            im[:,i,:,:,1] = z_final*PD_imag
            #  1 9 224 224 2
        # z_norm = np.linalg.norm(im_z[:,:,:,:,0])
        #self.T1map = T1map
        return im#*(1/z_norm)
   
    def forward(self, y): # |y - Ax|
        #print("-------> max of inp = ",np.linalg.norm(self.x_adj.detach().cpu().numpy()))
        #out_net =  self.network(self.x_adj/np.linalg.norm(self.x_adj.detach().cpu().numpy())) # output is T1, PD_Real, PD_imag (in that order)
        out_net =  self.network(self.x_adj/np.max(self.x_adj.detach().cpu().numpy()), self.T1map[None,None,...]/np.max(self.T1map.detach().cpu().numpy()))
        #print("-------> out_net.shape = ", out_net.shape)
        self.out_net = out_net
        out = self.signal_model(out_net[0,0,...],out_net[0,1,...],out_net[0,2,...])#covert out_net into complex image using signal equation
        np.save("/home/kalina/review/T1_pred.npy",out_net[0,0,...].detach().cpu().numpy())
        return out

    def get_metadata(self):
        return {}
