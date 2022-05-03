#!/usr/bin/env python


from test_tube import HyperOptArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#

from pytorch_lightning import seed_everything

import os
import pathlib
import argparse
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

import time

from deepinpy.recons import CGSenseRecon, MoDLRecon, ResNetRecon, DeepBasisPursuitRecon, CSDIPRecon

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import random # used to avoid race conditions, intentionall unseeded
import numpy.random

def main_train(args, gpu_ids=None):

    if args.hyperopt:
        time.sleep(random.random()) # used to avoid race conditions with parallel jobs
    tt_logger = TestTubeLogger(save_dir=args.logdir, name=args.name, debug=False, create_git_tag=False, version=args.version, log_graph=False)
    tt_logger.log_hyperparams(args)
    save_path = '{}/{}/version_{}'.format(args.logdir, tt_logger.name, tt_logger.version)
    args.save_path = save_path
    args.tt_logger_name = tt_logger.name
    args.tt_logger_version = tt_logger.version
    print('save path is', save_path)
    checkpoint_path = '{}/checkpoints'.format(save_path)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    if args.save_all_checkpoints:
        save_top_k = -1
    else:
        save_top_k = 1
    checkpoint_callback = ModelCheckpoint(checkpoint_path, 'epoch', save_top_k=save_top_k, mode='max', verbose=False)

    if args.recon == 'cgsense':
        MyRecon = CGSenseRecon
    elif args.recon == 'modl':
        MyRecon = MoDLRecon
    elif args.recon == 'resnet':
        MyRecon = ResNetRecon
    elif args.recon == 'dbp':
        MyRecon = DeepBasisPursuitRecon
    elif args.recon == 'csdip':
        MyRecon = CSDIPRecon

    M = MyRecon(args)

    if args.checkpoint_init:
        # FIXME: workaround for PyTL issues with loading hparams
        print('loading checkpoint: {}'.format(args.checkpoint_init))
        checkpoint = torch.load(args.checkpoint_init, map_location=lambda storage, loc: storage)
        M.load_state_dict(checkpoint['state_dict'])
    else:
        print('training from scratch')

    
    if gpu_ids is None:
        gpus = None
        distributed_backend = None # FIXME should this also be ddp?
    elif args.hyperopt:
        gpus = 1
        distributed_backend = None
    else:
        gpus = gpu_ids
        distributed_backend = 'ddp'


    trainer = Trainer(max_epochs=args.num_epochs, gpus=gpus, logger=tt_logger, checkpoint_callback=checkpoint_callback, distributed_backend=distributed_backend, accumulate_grad_batches=args.num_accumulate, progress_bar_refresh_rate=1, gradient_clip_val=args.clip_grads)

    trainer.fit(M)

if __name__ == '__main__':
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'deep inverse problems optimization'

    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='grid_search')

    parser.opt_list('--step', action='store', dest='step', type=float, tunable=True, options=[0.01], help='number of latent channels', default=0.0001)
    parser.opt_list('--latent_channels', action='store', dest='latent_channels', type=int, tunable=False, options=[64, 96, 128], help='number of latent channels', default=64)
    parser.opt_list('--num_blocks', action='store', dest='num_blocks', type=int, tunable=False, options=[4, 6, 8], help='number of ResNetBlocks', default=6)
    parser.opt_list('--z_dim', action='store', dest='z_dim', type=int, tunable=False, options=[2,4,8], help='number of z_dim channels', default=64)

    parser.opt_list('--continuation_scale1', action='store', type=int,  dest='continuation_scale1', tunable=False,options=[1],help='how much to scale mu1 by', default=1)
    parser.opt_list('--continuation_scale2', action='store', type=int,  dest='continuation_scale2', tunable=False,options=[0.1, 0.9, 0.95],help='how much to scale mu2 by', default=1)
    parser.opt_list('--continuation_rate', action='store', type=int,  dest='continuation_rate', tunable=False,options=[5000],help='when to update mu1 and mu2', default=10000)

    parser.opt_list('--mu1', action='store', type=int,  dest='mu1', tunable=True,options=[1],help='reg_rate for model consistency', default=0.5) #[0.05, 0.1, 0.5, 1]
    parser.opt_list('--mu2', action='store', type=int,  dest='mu2', tunable=False,options=[0],help='reg_rate for model change being gradual', default=0)
    parser.opt_list('--reg_update_freq', action='store', type=int, dest='reg_update_freq', tunable=True,options=[5],help='frequency with which to update model x_n', default=1)

    #parser.opt_range('--reg_rate', action='store',type=float, dest='reg_rate', help='set regularization rate', default=1,tunable=False, nb_samples=16, low=.001, high=5)
    #parser.opt_list('--reg_length', action='store', dest='reg_length', type=int, tunable=False, options=[200, 500, 1000,1500, 2000], help='number of z_dim channels', default=500)
    #parser.opt_list('--start_reg_epoch', action='store', type=int,  dest='start_reg_epoch', tunable=False,options=[500, 1000, 2000, 4000,5000],help='use regularization term after n epochs', default=5000)

    parser.opt_range('--l2lam_init', action='store', type=float, dest='l2lam_init', default=.001, tunable=False, low=.0001, high=100, help='initial l2 regularization')
    parser.opt_list('--solver', action='store', dest='solver', type=str, tunable=False, options=['sgd', 'adam', 'rmsprop'], help='optimizer/solver ("adam", "sgd", etc.)', default="adam")
    parser.opt_range('--cg_max_iter', action='store', dest='cg_max_iter', type=int, tunable=False, low=1, high=20, help='max number of conjgrad iterations', default=10)
    parser.opt_range('--batch_size', action='store', dest='batch_size', type=int, tunable=False, low=1, high=20, help='batch size', default=2)
    parser.opt_range('--num_unrolls', action='store', dest='num_unrolls', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of unrolls', default=4)
    parser.opt_range('--num_admm', action='store', dest='num_admm', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of ADMM iterations', default=3)
    parser.opt_list('--network', action='store', dest='network', type=str, tunable=False, options=['ResNet', 'ResNet5Block'], help='which denoiser network to use', default='ResNet')
    parser.opt_range('--dropout', action='store', dest='dropout', type=float, tunable=False, low=0., high=.5, help='dropout fraction', default=0.)
    parser.opt_list('--batch_norm', action='store_true', dest='batch_norm', tunable=False, options=[True, False], help='batch normalization', default=False)
    parser.opt_range('--clip_grads', action='store', type=float, dest='clip_grads', help='clip norm of gradient vector to val', default=0, tunable=False, nb_samples=10, low=0, high=500)

    parser.opt_list('--optimize_z', action='store_true', dest='optimize_z', tunable=False, help='optimize over generative model latent space', default=False)

    parser.add_argument('--num_accumulate', action='store', dest='num_accumulate', type=int, help='nunumber of batch accumulations', default=1)
    parser.add_argument('--name', action='store', dest='name', type=str, help='experiment name', default=1)
    parser.add_argument('--version', action='store', dest='version', type=int, help='version number', default=None)
    parser.add_argument('--gpu', action='store', dest='gpu', type=str, help='gpu number(s)', default=None)
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument('--recon', action='store', type=str, dest='recon', default='cgsense', help='reconstruction method')
    parser.add_argument('--data_file', action='store', dest='data_file', type=str, help='data.h5', default=None)
    parser.add_argument('--data_val_file', action='store', dest='data_val_file', type=str, help='data.h5', default=None)
    parser.add_argument('--num_data_sets', action='store', dest='num_data_sets', type=int, help='number of data sets to use', default=None)
    parser.add_argument('--num_workers', action='store', type=int,  dest='num_workers', help='number of workers', default=0)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    #parser.add_argument('--clip_grads', action='store', type=float, dest='clip_grads', help='clip norm of gradient vector to val', default=0)
    parser.add_argument('--cg_eps', action='store', type=float, dest='cg_eps', help='conjgrad eps', default=1e-4)
    parser.add_argument('--stdev', action='store', type=float, dest='stdev', help='complex valued noise standard deviation', default=0.)
    parser.add_argument('--max_norm_constraint', action='store', type=float, dest='max_norm_constraint', help='norm constraint on weights', default=None)
    parser.add_argument('--fully_sampled', action='store_true', dest='fully_sampled', help='fully_sampled', default=False)
    parser.add_argument('--adam_eps', action='store', type=float, dest='adam_eps', help='adam epsilon', default=1e-8)
    parser.add_argument('--inverse_crime', action='store_true', dest='inverse_crime', help='inverse crime', default=False)
    parser.add_argument('--use_sigpy', action='store_true', dest='use_sigpy', help='use SigPy for Linops', default=False)
    parser.add_argument('--noncart', action='store_true', dest='noncart', help='NonCartesian data', default=False)
    parser.add_argument('--abs_loss', action='store_true', dest='abs_loss', help='use magnitude for loss', default=False)
    parser.add_argument('--self_supervised', action='store_true', dest='self_supervised', help='self-supervised loss', default=False)
    parser.add_argument('--hyperopt', action='store_true', dest='hyperopt', help='perform hyperparam optimization', default=False)
    parser.add_argument('--checkpoint_init', action='store', dest='checkpoint_init', type=str, help='load from checkpoint', default=None)
    parser.add_argument('--logdir', action='store', dest='logdir', type=str, help='log dir', default='logs')
    parser.add_argument('--save_all_checkpoints', action='store_true', dest='save_all_checkpoints', help='Save all checkpoints', default=False)
    parser.add_argument('--lr_scheduler', action='store', dest='lr_scheduler', nargs='+', type=int, help='do [#epoch, learning rate multiplicative factor] to use a learning rate scheduler', default=-1)
    parser.add_argument('--save_every_N_epochs', action='store', type=int,  dest='save_every_N_epochs', help='save images every N epochs', default=1)
    parser.add_argument('--num_spatial_dimensions', action='store', dest='num_spatial_dimensions', type=int, help='num of spatial dimensions in ksp.shape, e.g. (..., Nx, Ny, Nz, ...) means 3 spatial dimensions. Currently 2D or 3D is supported', default=2)
    parser.add_argument('--do_t1_mapping', action='store_true', dest='do_t1_mapping', help='perform T1 mapping using loaded dictionary', default=False)
    parser.add_argument('--do_pk_mapping', action='store_true', dest='do_pk_mapping', help='perform T1 mapping using loaded dictionary', default=False)
    parser.add_argument('--save_img', action='store_true', dest='save_img', help='toggle on/off whether to save outputted images', default=False)
    parser.json_config('--config', default=None)
    parser.add_argument('--dictdir', action='store', dest='dictdir', type=str, help='points to dictionary files', default=None)
    parser.add_argument('--save_img_path', action='store', dest='save_img_path', type=str, help='points to where to sae images. default is save_path', default=None)

    args = parser.parse_args()


    torch.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic=True
    seed_everything(args.random_seed)

    if args.hyperopt:
        if args.gpu is None:
            args.optimize_parallel_cpu(main_train, nb_trials=args.num_trials, nb_workers=args.num_workers)
        else:
            gpu_ids = [a.strip() for a in args.gpu.split(',')]
            args.optimize_parallel_gpu(main_train, gpu_ids=gpu_ids, max_nb_trials=args.num_trials)
    else:
        if args.gpu is not None:
            gpu_ids = [int(a) for a in args.gpu.split(',')]
        else:
            gpu_ids = None
        main_train(args, gpu_ids=gpu_ids)
