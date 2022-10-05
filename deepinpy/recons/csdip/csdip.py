import sys

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import DCGAN_MRI, decodernw, convdecoder
from deepinpy.recons import Recon
import numpy as np
import torch


class CSDIPRecon(Recon):

    def __init__(self, args):
        super(CSDIPRecon, self).__init__(args)

        self.N1 = 8
        self.N2 = 8
        self.x_adj = None

        self.output_size = self.D.shape[1:]
        print('output size:', self.output_size)
        self.hparams.num_image_params = np.product(self.output_size) * 2 # real/imaginary

        if len(self.output_size) > 2:
            self.num_output_channels = 2 * np.prod(self.output_size[:-2])
            self.output_size = self.output_size[-2:]
        else:
            self.num_output_channels = 2

        if self.hparams.network == 'DCGAN':
            # FIXME: make work for arbitrary input sizes
            self.network = DCGAN_MRI(self.hparams.z_dim, ngf=64, output_size=self.output_size, nc=2, num_measurements=256)

        elif self.hparams.network == 'DeepDecoder':

            # initial number of channels given by z_dim
            self.num_channels_up = [self.hparams.z_dim] + [self.hparams.latent_channels]*(self.hparams.num_blocks - 1)

            scale_x = [int(np.product([self.N1] + [np.exp(np.log(self.output_size[0]/self.N1)/self.hparams.num_blocks)] * i)) for i in range(self.hparams.num_blocks)] + [self.output_size[0]]
            scale_y = [int(np.product([self.N2] + [np.exp(np.log(self.output_size[1]/self.N2)/self.hparams.num_blocks)] * i)) for i in range(self.hparams.num_blocks)] + [self.output_size[1]]

            self.upsample_size = list(zip(scale_x, scale_y))

            self.network = decodernw(num_output_channels=self.num_output_channels, num_channels_up=self.num_channels_up, upsample_first=True, need_sigmoid=False, upsample_size=self.upsample_size)
        
        elif self.hparams.network == 'ConvDecoder':
            
            self.N1, self.N2 = 16, 16
            self.in_size = [self.N1, self.N2]

            self.network = convdecoder(num_output_channels=self.num_output_channels, strides=[1]*self.hparams.num_blocks, out_size=self.output_size, in_size=self.in_size, num_channels=self.hparams.latent_channels, z_dim=self.hparams.z_dim, num_layers=self.hparams.num_blocks, need_sigmoid=False, upsample_mode='nearest', skips=False, need_last=True)

        else:
            # FIXME: error logging
            print('ERROR: invalid network specified')
            sys.exit(-1)
        # self.network gets defined above, so at this point you can do self.network.load_state_dict(torch.load(PATH),strict=False).
        # Need an if-statement based on a flag that says "yes warmstart"
        if self.hparams.do_warmstart:
            self.network.load_state_dict(torch.load(self.hparams.state_dict_path),strict=False)
        self.zseed = None # to be initialized on first batch
        self.A = None # to be initialized on first batch

        # save number of image and model params to the hyperparameters struct
        self.hparams.num_network_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.hparams.compression_factor = self.hparams.num_image_params / self.hparams.num_network_params

        self.optimize_z = self.hparams.optimize_z

        if self.hparams.network == 'ConvDecoder':
            zseed = torch.zeros(len(self.D), self.hparams.z_dim, self.N1, self.N2) 
        elif self.hparams.network == 'DeepDecoder':
            zseed = torch.zeros(len(self.D), self.hparams.z_dim, self.N1, self.N2) 
        elif self.hparams.network == 'DCGAN':
            print('Warning: DCGAN no longer supported')
            zseed = torch.zeros(len(self.D)*self.hparams.z_dim).view(self.hparams.batch_size,self.hparams.z_dim,1,1)

        zseed.data.normal_().type(torch.FloatTensor)
        self.zseed = zseed


        if self.optimize_z:
            self.zseed = torch.nn.Parameter(self.zseed)

    def batch(self, data):
        maps = data['maps']
        masks = data['masks']
        inp = data['out'] #read in maps, masks, and k-space input

        # initialize z vector only once per index
        if not self.optimize_z:
            self.zseed = self.zseed.to(inp.device)

        self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)

        self.batch_idx = data['idx']

    def forward(self, y):
        zseed = self.zseed[self.batch_idx,...]
        if self.hparams.batch_size == 1:
            zseed = zseed[None,...]
        out =  self.network(zseed) #DCGAN acts on the low-dim space parameterized by z to output the image x
        if (self.hparams.network == 'DeepDecoder') or (self.hparams.network == 'ConvDecoder'):
            if len(self.D.shape) == 4:
                out = out.reshape(out.shape[0], 2, -1, out.shape[-2], out.shape[-1])
                out = out.permute((0, 2, 3, 4, 1))
            else:
                out = out.permute(0, 2, 3, 1)
        PATH = "{0}/{1}_version{2}_epoch{3}_state_dict.pt".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch)
        if (self.current_epoch % self.hparams.save_every_N_epochs == 0 or self.current_epoch == self.hparams.num_epochs - 1):
            torch.save(self.network.state_dict(), PATH)
        return out

    def get_metadata(self):
        return {}
