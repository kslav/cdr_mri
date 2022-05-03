"""Recon object for combining system blocks (such as datasets and transformers),
model blocks (such as CNNs and ResNets), and optimization blocks (such as conjugate
gradient descent)."""

#!/usr/bin/env python

import numpy as np
import torch
import sys

import pytorch_lightning as pl
#from pytorch_lightning.utilities import grad_norm

from deepinpy.utils import utils
from deepinpy.utils import T1_mapping
from deepinpy.utils import PK_mapping
from deepinpy import opt
import deepinpy.utils.complex as cp
from deepinpy.forwards import MultiChannelMRIDataset
from deepinpy.forwards import maps_adj
from deepinpy.forwards import maps_forw

from torchvision.utils import make_grid
from torch.optim import lr_scheduler 

@torch.jit.script
def calc_nrmse(gt, pred):
    return (opt.ip_batch(pred - gt) / opt.ip_batch(gt)).sqrt().mean()


class Recon(pl.LightningModule):
    """An abstract class for implementing system-model-optimization (SMO) constructions.

    The Recon is an abstract class which outlines common functionality for all SMO structure implementations. All of them share hyperparameter initialization, MCMRI dataset processing and loading, loss function, training step, and optimizer code. Each implementation of Recon must provide batch, forward, and get_metadata methods in order to define how batches are created from the data, how the model performs its forward pass, and what metadata the user should be able to return. Currently, Recon automatically builds the dataset as an MultiChannelMRIDataset object; overload _build_data to circumvent this.

    Args:
        hprams (dict): Key-value pairings with parameter names as keys.

    Attributes:
        hprams (dict): Key-value pairings with hyperparameter names as keys.
        _loss_fun (func): Set to use either torch.nn.MSELoss or _abs_loss_fun.
        D (MultiChannelMRIDataset): Holds the MCMRI dataset.

    """

    def __init__(self, hparams):
        super(Recon, self).__init__()

        self._init_hparams(hparams)
        self._build_data()
        self.scheduler = None
        self.log_dict = None

    def _init_hparams(self, hparams):
        self.hparams = hparams

        self._loss_fun = torch.nn.MSELoss(reduction='sum')

        if self.hparams.abs_loss:
            self.loss_fun = self._abs_loss_fun
        elif self.hparams.do_t1_mapping:
            self.loss_fun = self._t1_mapping_loss_fun_v3
        elif self.hparams.do_pk_mapping:
            self.loss_fun = self._pk_mapping_loss_fun
        else:
            self.loss_fun = self._loss_fun
        if self.hparams.save_img_path == None:
            self.hparams.save_img_path = self.hparams.save_path


    def _build_data(self):
        self.D = MultiChannelMRIDataset(data_file=self.hparams.data_file, stdev=self.hparams.stdev, num_data_sets=self.hparams.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, scale_data=False, fully_sampled=self.hparams.fully_sampled, data_idx=None, inverse_crime=self.hparams.inverse_crime, noncart=self.hparams.noncart)

    def _abs_loss_fun(self, x_hat, imgs):
        x_hat_abs = torch.sqrt(x_hat.pow(2).sum(dim=-1))
        imgs_abs = torch.sqrt(imgs.pow(2).sum(dim=-1))
        return self._loss_fun(x_hat_abs, imgs_abs)

    # def _t1_mapping_loss_fun(self, x_hat, imgs):#,reg_rate):
    #     # make numpy copy of x_hat
    #     #all of this stuff would be under the mod(conitional epoch_cutoff,curr_epoch)
    #     x_hat_copy = x_hat
    #     #x_hat_abs_test = cp.zabs(x_hat)
    #     x_hat_abs = torch.sqrt(x_hat_copy.pow(2).sum(dim=-1))[0,...]
    #     print("x_hat_abs.shape = ", x_hat_abs.shape)
    #     inp_data = x_hat
    #     print("x_hat.type = ", x_hat.type)
    #     if self.hparams.self_supervised:
    #         print("self_supervised = ON")
    #         inp_data =  self.A.forward(x_hat)

    #     print("x_hat_abs.type", x_hat_abs.type)

    #     #print("x_hat_model.size() = ", x_hat_model.size())
    #     reg_scalar = self.hparams.reg_rate
    #     #if self.current_epoch % self.hparams.start_reg_epoch <= self.hparams.reg_length :
    #     if self.current_epoch % self.hparams.reg_rate_mod == 0:
    #         print("current_epoch = ", self.current_epoch)
    #         reg_scalar = self.hparams.reg_rate_scale * reg_scalar
    #         self.hparams.reg_rate = reg_scalar
    #         print("reg_rate is now ", self.hparams.reg_rate)
    #     with torch.no_grad():
    #         #D = (torch.load("/home/kalina/Code/deepinpy_private/modelling/dict_T1mapping_FAs_5_10_15_20_N2000.pt")).to(x_hat.device) #send to same device as x_hat
    #         #sc_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/sc_T1mapping_FAs_5_10_15_20_N2000.pt")).to(x_hat.device) # repeat same steps as D
    #         #t1_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/t1_list_50_4000_2k.pt")).to(x_hat.device) # see above

    #         D = (torch.load("/home/kalina/Code/deepinpy_private/modelling/dict_T1mapping_TR_7p6_FAs_3_5_10_15_v2.pt")).to(x_hat.device) #send to same device as x_hat
    #         sc_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/sc_T1mapping_TR_7p6_FAs_3_5_10_15_v2.pt")).to(x_hat.device) # repeat same steps as D
    #         t1_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/t1_list_1_4000_15k_v2.pt")).to(x_hat.device) # see above

    #         sc_out, idx_out, model_out = T1_mapping.dictionary_match(x_hat_abs, D, t1_list, sc_list) #line 97 - 103 put it under with torch.no_grad()
    #         #sc_out_gt, idx_out_gt, gt_model = T1_mapping.dictionary_match(gt_abs, D, t1_list, sc_list)
    #         #torch.save(idx_out,'/home/kalina/MDA_MFA_idx_out_debug_epoch2002.pt')
    #         #torch.save(x_hat_model, '/home/kalina/MDA_MFA_x_model_out_debug_epoch2002.pt')
    #         #torch.save(idx_out_gt,'/home/kalina/MDA_MFA_idx_out_gt.pt') #save as numpy array
    #         #torch.save(gt_model, '/home/kalina/MDA_MFA_gt_model_out_debug_epoch2002.pt')
    #         #torch.save(sc_out_gt, '/home/kalina/MDA_MFA_sc_out_gt.pt')
    #     #print("model_out.shape = ",model_out.shape)
    #     x_hat_abs_cp = torch.stack((x_hat_abs,torch.zeros(x_hat_abs.shape).to(x_hat_abs.device)), 3)[None,...]
    #     model_out_cp = torch.stack((model_out,torch.zeros(model_out.shape).to(x_hat_abs.device)), 3)[None,...]
    #     print("x_hat_abs_cp.shape = ", x_hat_abs_cp.shape)
    #     print("model_out_cp.shape = ", model_out_cp.shape)
    #     loss_t1_mapping = torch.nn.MSELoss(reduction='sum')
    #     print("currently, reg_scalar = ", reg_scalar)
    #     return self._loss_fun(inp_data, imgs) + reg_scalar*loss_t1_mapping(model_out_cp.float(),x_hat_abs_cp.float()) #don't compute it a second time 
    #         #return self._loss_fun(inp, imgs) + reg_scalar*loss_t1_mapping(torch.zeros(x_hat.shape).to(x_hat.device),x_hat_abs_test)
    #     #else:
    #         #return self._loss_fun(inp_data,imgs)
    # def _t1_mapping_loss_fun_v2(self, x_hat, imgs):#,reg_rate):
    #     # make numpy copy of x_hat
    #     #all of this stuff would be under the mod(conitional epoch_cutoff,curr_epoch)
    #     x_hat_copy = x_hat
    #     #x_hat_abs_test = cp.zabs(x_hat)
    #     x_hat_abs = torch.sqrt(x_hat_copy.pow(2).sum(dim=-1))[0,...]
    #     print("x_hat_abs.shape = ", x_hat_abs.shape)
    #     inp_data = x_hat
    #     print("x_hat.type = ", x_hat.type)
    #     if self.hparams.self_supervised:
    #         print("self_supervised = ON")
    #         inp_data =  self.A.forward(x_hat)

    #     print("x_hat_abs.type", x_hat_abs.type)

    #     #print("x_hat_model.size() = ", x_hat_model.size())
    #     reg_scalar = self.hparams.reg_rate
    #     #if self.current_epoch % self.hparams.start_reg_epoch <= self.hparams.reg_length :

    #     if self.current_epoch % self.hparams.reg_rate_mod == 0:
    #         print("current_epoch = ", self.current_epoch)
    #         reg_scalar = self.hparams.reg_rate_scale * reg_scalar
    #         self.hparams.reg_rate = reg_scalar
    #         print("reg_rate is now ", self.hparams.reg_rate)
    #     with torch.no_grad():
    #         #D = (torch.load("/home/kalina/Code/deepinpy_private/modelling/dict_T1mapping_FAs_5_10_15_20_N2000.pt")).to(x_hat.device) #send to same device as x_hat
    #         #sc_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/sc_T1mapping_FAs_5_10_15_20_N2000.pt")).to(x_hat.device) # repeat same steps as D
    #         #t1_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/t1_list_50_4000_2k.pt")).to(x_hat.device) # see above

    #         D = (torch.load("/home/kalina/Code/deepinpy_private/modelling/dict_T1mapping_TR_7p6_FAs_3_5_10_15_v2.pt")).to(x_hat.device) #send to same device as x_hat
    #         sc_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/sc_T1mapping_TR_7p6_FAs_3_5_10_15_v2.pt")).to(x_hat.device) # repeat same steps as D
    #         t1_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/t1_list_1_4000_15k_v2.pt")).to(x_hat.device) # see above

    #         sc_out, idx_out, model_out = T1_mapping.dictionary_match(x_hat_abs, D, t1_list, sc_list) #line 97 - 103 put it under with torch.no_grad()
    #         #sc_out_gt, idx_out_gt, gt_model = T1_mapping.dictionary_match(gt_abs, D, t1_list, sc_list)
    #         #torch.save(idx_out,'/home/kalina/MDA_MFA_idx_out_debug_epoch2002.pt')
    #         #torch.save(x_hat_model, '/home/kalina/MDA_MFA_x_model_out_debug_epoch2002.pt')
    #         #torch.save(idx_out_gt,'/home/kalina/MDA_MFA_idx_out_gt.pt') #save as numpy array
    #         #torch.save(gt_model, '/home/kalina/MDA_MFA_gt_model_out_debug_epoch2002.pt')
    #         #torch.save(sc_out_gt, '/home/kalina/MDA_MFA_sc_out_gt.pt')
    #     #print("model_out.shape = ",model_out.shape)
    #     x_hat_abs_cp = torch.stack((x_hat_abs,torch.zeros(x_hat_abs.shape).to(x_hat_abs.device)), 3)[None,...]
    #     model_out_cp = torch.stack((model_out,torch.zeros(model_out.shape).to(x_hat_abs.device)), 3)[None,...]
    #     print("x_hat_abs_cp.shape = ", x_hat_abs_cp.shape)
    #     print("model_out_cp.shape = ", model_out_cp.shape)
    #     loss_t1_mapping = torch.nn.MSELoss(reduction='sum')
    #     print("currently, reg_scalar = ", reg_scalar)
    #     return self._loss_fun(inp_data, imgs) + reg_scalar*loss_t1_mapping(model_out_cp.float(),x_hat_abs_cp.float()) #don't compute it a second time 
    
    def _t1_mapping_loss_fun_v3(self, x_hat, imgs):#,reg_rate):
        # make numpy copy of x_hat
        x_hat_copy = x_hat
        #get the magnitude image x_hat_abs
        print("x_hat_copy.shape is ", x_hat_copy.shape)
        x_hat_abs = torch.sqrt(x_hat_copy.pow(2).sum(dim=-1))[0,...]
        print("x_hat_abs.shape = ", x_hat_abs.shape)
        print("x_hat.type = ", x_hat.type)
        inp_data = x_hat
        # if self_supervised, we need to find k-space of x_hat using forward operator!
        if self.hparams.self_supervised:
            print("self_supervised = ON")
            inp_data =  self.A.forward(x_hat)

        print("x_hat_abs.type", x_hat_abs.type)

        with torch.no_grad():
            #load the dictionary (D), scale factors (sc_list), and t1_list
            #self.D instead
            if self.current_epoch == 0:
                #self.model_dict = (torch.load("/home/kalina/Code/deepinpy_private_201208/deepinpy/utils/dict_T1mapping_TR_6_FAs_2-20_num10.pt")).to(x_hat.device) #send to same device as x_hat
                #self.sc_list = (torch.load("/home/kalina/Code/deepinpy_private_201208/deepinpy/utils/sc_T1mapping_TR_6_FAs_2-20_num10.pt")).to(x_hat.device) # repeat same steps as D
                #self.t1_list = (torch.load("/home/kalina/Code/deepinpy_private_201208/deepinpy/utils/t1_list_1_4000_10k.pt")).to(x_hat.device) # see above
                self.model_dict = (torch.load("{}/dict_T1mapping_TR_6_FAs_4-20_2k.pt".format(self.hparams.dictdir))).to(x_hat.device) #send to same device as x_hat
                self.sc_list    = (torch.load("{}/sc_T1mapping_TR_6_FAs_4-20_2k.pt".format(self.hparams.dictdir))).to(x_hat.device) # repeat same steps as D
                self.t1_list    = (torch.load("{}/t1_list_50_4000_2k.pt".format(self.hparams.dictdir))).to(x_hat.device) # see above
                #torch.save(imgs, '/home/kalina/x_gt_phantom_t1regv3_12052021.pt')
            #update x_i via update rule DF[/mu * G(w_i) + /mu' * x_i-1] where x_i-1 = self.model_out
            x_mp = self.x_m
            self.model_out_cp = x_mp
            self.x_mp_cp = x_mp
            mu1 = self.hparams.mu1
            mu2 = self.hparams.mu2
            print("mu1 and mu2 are ", mu1, " and ", mu2)
            if (self.current_epoch % self.hparams.continuation_rate == 0) and self.current_epoch != 0:
                print("current_epoch = ", self.current_epoch)
                mu1 = (self.hparams.continuation_scale1) * self.hparams.mu1
                mu2 = (self.hparams.continuation_scale2) * self.hparams.mu2
                self.hparams.mu1 = mu1
                self.hparams.mu2 = mu2
            #Do dictionary matching every N epochs (reg_update_freq) to update the regularization term ()
            if (self.current_epoch % self.hparams.reg_update_freq == 0) and self.current_epoch != 0:
                print("updating model...epoch num. = ", self.current_epoch, "reg_update_freq = ", self.hparams.reg_update_freq)
                D = self.model_dict
                t1_list = self.t1_list
                sc_list = self.sc_list
                #if mu2 != 0:
                #   sc_out, idx_out, model_out = T1_mapping.dictionary_match((mu1*x_hat_abs + mu2*x_mp), D, t1_list, sc_list)
                #else:
                    #if self.current_epoch == 0:
                     #   print("confirming mu2 = 0 and we're using this version of regularization")
                sc_out, idx_out, model_out = T1_mapping.dictionary_match(x_hat_abs, D, t1_list, sc_list)
                model_out_cp = torch.stack((model_out,torch.zeros(model_out.shape).to(x_hat_abs.device)), 3)[None,...]
                x_mp_cp = torch.stack((x_mp,torch.zeros(x_mp.shape).to(x_hat_abs.device)), 3)[None,...]
                #update self.x_m to be model_out
                if self.current_epoch != 0:
                    self.x_m = model_out
                    self.model_out_cp = model_out_cp
                    self.x_mp_cp = x_mp_cp
            #save the reconstruce image at the final epoch so you don't take up computing power
            #if (self.current_epoch % self.hparams.save_every_N_epochs == 0): 
            #    torch.save(model_out, "/home/kalina/{0}_version{1}_epoch{2}_x_m.npy".format(self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch))
            #    #torch.save(x_hat, '{}/x_hat.pt'.format(self.hparams.save_path))
            

        #cp stands for complex, model_out_cp is the updated x_i
        x_hat_abs_cp = x_hat_abs#torch.stack((x_hat_abs,torch.zeros(x_hat_abs.shape).to(x_hat_abs.device)), 3)[None,...]
        model_out_cp_v2 = self.x_m
        print("x_hat_abs_cp.shape = ", x_hat_abs_cp.shape)
        print("model_out.shape = ", model_out_cp_v2.shape)
        #first regularization term is ||G(w) - x_i||
        reg1 = torch.nn.MSELoss(reduction='sum')
        #second regularization term is ||x_i - x_i-1
        reg2 = torch.nn.MSELoss(reduction='sum')
        #self.reg1 = reg1 
        self.reg1 = mu1*reg1(model_out_cp_v2.float(),x_hat_abs_cp.float())
        self.reg2 = mu2*reg2(self.model_out_cp.float(), self.x_mp_cp.float())
        self.loss_data = self._loss_fun(inp_data, imgs)
        
        return self._loss_fun(inp_data, imgs) + mu1*reg1(model_out_cp_v2.float(),x_hat_abs_cp.float()) + mu2*reg2(self.model_out_cp.float(), self.x_mp_cp.float())
    
    # def _pk_mapping_loss_fun(self, x_hat, imgs):#,reg_rate):
    #     # make numpy copy of x_hat
    #     #all of this stuff would be under the mod(conitional epoch_cutoff,curr_epoch)
    #     x_hat_copy = x_hat
    #     #x_hat_abs_test = cp.zabs(x_hat)
    #     x_hat_abs = torch.sqrt(x_hat_copy.pow(2).sum(dim=-1))[0,...]
    #     print("x_hat_abs.shape = ", x_hat_abs.shape)
    #     inp_data = x_hat
    #     print("x_hat.type = ", x_hat.type)
    #     if self.hparams.self_supervised:
    #         print("self_supervised = ON")
    #         inp_data =  self.A.forward(x_hat)

    #     print("x_hat_abs.type", x_hat_abs.type)

    #     #print("x_hat_model.size() = ", x_hat_model.size())
    #     reg_scalar = 0
    #     if self.current_epoch % self.hparams.start_reg_epoch <= self.hparams.reg_length :
    #         print("current_epoch mod start_epoch = ", (self.current_epoch % self.hparams.start_reg_epoch))
    #         reg_scalar = self.hparams.reg_rate
    #         with torch.no_grad():
    #             D = (torch.load("/home/kalina/Code/deepinpy_private/modelling/dict_PKmapping_TR_7p6_FAs_3_5_10_15_v2.pt")).to(x_hat.device) #send to same device as x_hat
    #             sc_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/sc_PKmapping_TR_7p6_FAs_3_5_10_15_v2.pt")).to(x_hat.device) # repeat same steps as D
    #             t1_list = (torch.load("/home/kalina/Code/deepinpy_private/modelling/t1_list_1_4000_15k_15k_v2.pt")).to(x_hat.device) # see above

    #             sc_out, idx_out, model_out = PK_mapping.dictionary_match(x_hat_abs, D, t1_list, sc_list) #line 97 - 103 put it under with torch.no_grad()
    #             #sc_out_gt, idx_out_gt, gt_model = T1_mapping.dictionary_match(gt_abs, D, t1_list, sc_list)
    #             #torch.save(idx_out,'/home/kalina/MDA_MFA_idx_out_debug_epoch2002.pt')
    #             #torch.save(x_hat_model, '/home/kalina/MDA_MFA_x_model_out_debug_epoch2002.pt')
    #             #torch.save(idx_out_gt,'/home/kalina/MDA_MFA_idx_out_gt.pt') #save as numpy array
    #             #torch.save(gt_model, '/home/kalina/MDA_MFA_gt_model_out_debug_epoch2002.pt')
    #             #torch.save(sc_out_gt, '/home/kalina/MDA_MFA_sc_out_gt.pt')
    #         #print("model_out.shape = ",model_out.shape)
    #         x_hat_abs_cp = torch.stack((x_hat_abs,torch.zeros(x_hat_abs.shape).to(x_hat_abs.device)), 3)[None,...]
    #         model_out_cp = torch.stack((model_out,torch.zeros(model_out.shape).to(x_hat_abs.device)), 3)[None,...]
    #         print("x_hat_abs_cp.shape = ", x_hat_abs_cp.shape)
    #         print("model_out_cp.shape = ", model_out_cp.shape)
    #         loss_t1_mapping = torch.nn.MSELoss(reduction='sum')
    #         print("currently, reg_scalar = ", reg_scalar)
    #         return self._loss_fun(inp_data, imgs) + reg_scalar*loss_t1_mapping(model_out_cp.float(),x_hat_abs_cp.float()) #don't compute it a second time 
    #         #return self._loss_fun(inp, imgs) + reg_scalar*loss_t1_mapping(torch.zeros(x_hat.shape).to(x_hat.device),x_hat_abs_test)
    #     else:
    #         return self._loss_fun(inp_data,imgs)

    def batch(self, data):
        """Not implemented, should define a forward operator A and the adjoint matrix of the input x.

        Args:
            data (Tensor): The data which the batch will be drawn from.

        Raises:
        	NotImplementedError: Method needs to be implemented.
        """

        raise NotImplementedError

    def forward(self, y):
        """Not implemented, should perform a prediction using the implemented model.

        Args:
        	y (Tensor): The data which will be passed to the model for processing.

        Returns:
            The model’s prediction in Tensor form.

        Raises:
        	NotImplementedError: Method needs to be implemented.
        """

    def get_metadata(self):
        """Accesses metadata for the Recon.

        Returns:
            A dict holding the Recon’s metadata.

        Raises:
        	NotImplementedError: Method needs to be implemented.
        """
        raise NotImplementedError

    # FIXME: batch_nb parameter appears unused.
    def training_step(self, batch, batch_idx):
        """Defines a training step solving deep inverse problems, including batching, performing a forward pass through
        the model, and logging data. This may either be supervised or unsupervised based on hyperparameters.

        Args:
            batch (tuple): Should hold the indices of data and the corresponding data, in said order.
            batch_idx (None): Index of current batch

        Returns:
            A dict holding performance data and current epoch for performance tracking over time.
        """

        idx, data = batch
        idx = utils.itemize(idx)
        imgs = data['imgs']
        inp = data['out']
        maps = data['maps']
        #torch.save(inp,"{0}/{1}/lc{2}_zdim{3}_nb{4}_step{5}_inp.pt".format(self.hparams.logdir,self.hparams.name,self.hparams.latent_channels,self.hparams.z_dim,self.hparams.num_blocks,self.hparams.step))
        data['idx'] = idx
        
        self.batch(data)

        x_hat = self.forward(inp)

        try:
            num_cg = self.get_metadata()['num_cg']
        except KeyError:
            num_cg = 0

        if self.current_epoch == 0:
            x_adj = self.A.adjoint(inp)
            self.x_m = torch.sqrt(x_adj.pow(2).sum(dim=-1))[0,...]  #x_adj = self.A.adjoint(inp), this line would just be x_m = x_adj

        if self.logger and (self.current_epoch % self.hparams.save_every_N_epochs == 0 or self.current_epoch == self.hparams.num_epochs - 1):
            _b = inp.shape[0]
            if _b == 1 and idx == 0:
                    _idx = 0
            elif _b > 1 and 0 in idx:
                _idx = idx.index(0)
            else:
                _idx = None
            if _idx is not None:
                with torch.no_grad():
                    if self.x_adj is None:
                        x_adj = self.A.adjoint(inp)
                    else:
                        x_adj = self.x_adj

                    _x_hat = utils.t2n(x_hat[_idx,...]) 
                    _x_gt = utils.t2n(imgs[_idx,...])
                    _x_adj = utils.t2n(x_adj[_idx,...])
                    _x_m = self.x_m.cpu().numpy()
                    if self.hparams.self_supervised:
                        print("applying StS operation to x_hat...")
                        x_hat_StS = maps_adj(maps_forw(x_hat, maps),maps)
                        _x_hat = utils.t2n(x_hat_StS[_idx,...])
                        print("self.hparams.save_img = ", self.hparams.save_img)
                        if self.hparams.save_img:

                            #np.save("/home/kalina/images/Sub5_raw_images/{0}_{1}_x_gt.npy".format(self.hparams.name,self.hparams.version), _x_gt)
                            #np.save("{0}/x_hat.npy".format(self.hparams.save_path), _x_hat)
                            #np.save("/home/kalina/images/Sub5_raw_images/{0}_version{1}_epoch{2}_x_hat.npy".format(self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _x_hat)
                            #np.save("/home/kalina/images/Sub5_raw_images/{0}_version{1}_epoch{2}_x_m.npy".format(self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _x_hat)
                            np.save("{0}/{1}_version{2}_epoch{3}_x_hat.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _x_hat)
                            np.save("{0}/{1}_version{2}_epoch{3}_x_m.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _x_hat)

                    if len(_x_hat.shape) > 2:
                        #_d = tuple(range(len(_x_hat.shape)-2))
                        _d = _x_hat.shape[0]
                        _x_gt_dyn = np.abs(_x_gt)
                        #myim = torch.tensor(np.stack((_x_hat_fa1, _x_hat_fa2, _x_hat_fa3,_x_hat_fa4), axis=0))[:, None, ...] 
                        myim = torch.tensor(np.abs(_x_hat))[:, None, ...] 
                        grid = make_grid(myim, scale_each=True, normalize=True, nrow=_d, pad_value=10)
                        self.logger.experiment.add_image('4_train_prediction_dyn', grid, self.current_epoch)

                        mymod = self.x_m[:,None,...]
                        grid = make_grid(mymod, scale_each=True, normalize=True, nrow=_d, pad_value=10)
                        self.logger.experiment.add_image('7_phys_model_dyn', grid, self.current_epoch)



                        #myim = torch.tensor(np.stack((_x_gt_fa1, _x_gt_fa2, _x_gt_fa3,_x_gt_fa4), axis=0))[:, None, ...] 
                        #myim = torch.tensor(_x_gt_img_all)[:, None, ...] 
                        #grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                        #self.logger.experiment.add_image('5_ground_truth_dyn', grid, self.current_epoch)

                        #_x_hat_rss = np.linalg.norm(_x_hat, axis=_d)
                        #_x_gt_rss = np.linalg.norm(_x_gt, axis=_d)
                        #_x_adj_rss = np.linalg.norm(_x_adj, axis=_d)

                        #myim = torch.tensor(np.stack((_x_adj_rss, _x_hat_rss, _x_gt_rss), axis=0))[:, None, ...] 
                        #grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                        #self.logger.experiment.add_image('3_train_prediction_rss', grid, self.current_epoch)
              
                        while len(_x_hat.shape) > 2:
                            _x_hat = _x_hat[0,...]
                            _x_gt = _x_gt[0,...]
                            _x_adj = _x_adj[0,...]


                    myim = torch.tensor(np.stack((np.abs(_x_hat), np.angle(_x_hat)), axis=0))[:, None, ...] 
                    grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                    self.logger.experiment.add_image('2_train_prediction', grid, self.current_epoch)

                    if self.current_epoch == 0:
                            myim = torch.tensor(np.stack((np.abs(_x_gt), np.angle(_x_gt)), axis=0))[:, None, ...]
                            grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                            self.logger.experiment.add_image('1_ground_truth', grid, 0)

                            myim = torch.tensor(np.stack((np.abs(_x_adj), np.angle(_x_adj)), axis=0))[:, None, ...]
                            grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                            self.logger.experiment.add_image('0_input', grid, 0)

                            #myim = torch.tensor(_x_gt_dyn)[:, None, ...] 
                            #grid = make_grid(myim, scale_each=True, normalize=True, nrow=_d, pad_value=10)
                            #self.logger.experiment.add_image('5_ground_truth_dyn', grid, self.current_epoch)
        _reg1 = 0
        _reg2 = 0
        _loss_data = 0
        if self.hparams.self_supervised:
            pred = self.A.forward(x_hat)
            gt = inp
        else:
            pred = x_hat
            gt = imgs

        if self.hparams.self_supervised and self.hparams.do_t1_mapping:
            loss = self.loss_fun(x_hat, gt) 
            _reg1_prelim = self.reg1
            _reg2_prelim = self.reg2
            _loss_data_prelim = self.loss_fun(x_hat,gt)
            _reg1 = _reg1_prelim.clone().detach().requires_grad_(False)
            _reg2 = _reg2_prelim.clone().detach().requires_grad_(False)
            _loss_data = _loss_data_prelim.clone().detach().requires_grad_(False)
        else:
            loss = self.loss_fun(pred, gt)

        _loss = loss.clone().detach().requires_grad_(False)

        try:
            _lambda = self.l2lam.clone().detach().requires_grad_(False)
        except:
            _lambda = 0
        _epoch = self.current_epoch
        #_nrmse = calc_nrmse(imgs, x_hat).detach().requires_grad_(False)
        imgs_nrmse = maps_adj(maps_forw(imgs, maps),maps)
        x_hat_nrmse = maps_adj(maps_forw(x_hat, maps),maps)
        _nrmse = calc_nrmse(imgs_nrmse, x_hat_nrmse).detach().requires_grad_(False)
        #print("imgs.shape = ",imgs_nrmse.shape)
        #print("x_hat_nrmse.shape = ", x_hat_nrmse.shape)
        _num_cg = np.max(num_cg)

        #compute the total gradient here:
        _total_norm = 0
        #for p in self.network.parameters():
        #    if p.grad is not None:
        #        param_norm = p.grad.data.norm(2)
        #        _total_norm += param_norm.item() ** 2
        #        _total_norm = _total_norm ** 0.5
        #_grad_norm_total = _total_norm
        #parameters = [p for p in self.network.parameters() if p.grad is not None and p.requires_grad]
        #if len(parameters) == 0:
        #    total_norm = 0.0
        #else:
        #    device = parameters[0].grad.device
        #    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
        #grad_norms_all = grad_norm(self.network,2)
        #_grad_norm_total = grad_norms_all["grad_2_norm_total"]

        log_dict = {
                '1_lambda': _lambda,
                '4_train_loss': _loss,
                '0_epoch': self.current_epoch,
                '8_nrmse': _nrmse, 
                '2_max_num_cg': _num_cg,
                '3_val_loss': 0.,
                '6_t1_reg_loss_mu1': _reg1,
                '7_t1_reg_loss_mu2': _reg2,
                '5_data_consistency': _loss_data
                #'8_grad_norm_total': _grad_norm_total
                #'reg1': self.reg1
                }
        if self.current_epoch == self.hparams.num_epochs - 1:
            np.save("{0}/{1}_version{2}_epoch{3}_loss.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _loss.cpu())
            np.save("{0}/{1}_version{2}_epoch{3}_nrmse.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _nrmse.cpu())
            if self.hparams.do_t1_mapping:
                np.save("{0}/{1}_version{2}_epoch{3}_reg1.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _reg1.cpu())
                np.save("{0}/{1}_version{2}_epoch{3}_data_consist.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _loss_data.cpu())
        
        if self.logger:
            for key in log_dict.keys():
                self.logger.experiment.add_scalar(key, log_dict[key], self.global_step)

        self.log_dict = log_dict

        return loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        if self.log_dict:
            for key in self.log_dict.keys():
                if type(self.log_dict[key]) == torch.Tensor:
                    items[key] = utils.itemize(self.log_dict[key])
                else:
                    items[key] = self.log_dict[key]
        return items

    def configure_optimizers(self):
        """Determines whether to use Adam or SGD depending on hyperparameters.

        Returns:
            Torch’s implementation of SGD or Adam, depending on hyperparameters.
        """

        if 'adam' in self.hparams.solver:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.step)
        elif 'sgd' in self.hparams.solver:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.step)
        elif 'rmsprop' in self.hparams.solver:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.step) 
        if(self.hparams.lr_scheduler != -1):
            # doing self.scheduler will create a scheduler instance in our self object
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.lr_scheduler[0], gamma=self.hparams.lr_scheduler[1])
        if self.scheduler is None:
            return [self.optimizer]
        else:                
            return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        """Creates a DataLoader object, with distributed training if specified in the hyperparameters.

        Returns:
            A PyTorch DataLoader that has been configured according to the hyperparameters.
        """

        return torch.utils.data.DataLoader(self.D, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle, num_workers=0, drop_last=True)
