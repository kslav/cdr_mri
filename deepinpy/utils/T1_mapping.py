#code for making the dictionary
import numpy as np
import time
import joblib
import sys
sys.path.append("/home/kalina/Code/deepinpy_private_201208/")
import deepinpy.utils.complex as cp
import torch
#import module utils

def make_T1_list(T1_min,T1_max,N_T1):
    T1_list = np.linspace(T1_min, T1_max, N_T1)
    print(T1_list.shape)
    return T1_list[...,None]
def make_B1_list(B1_min,B1_max,N_B1):
    B1_list = np.linspace(B1_min, B1_max, N_B1)
    print(B1_list.shape)
    return B1_list[...,None]

def make_T1B1_list(T1_list,B1_list):
    xind,yind = np.meshgrid(T1_list,B1_list)
    T1B1_list = np.vstack([xind.ravel(),yind.ravel()])
    return T1B1_list

def make_T1_dict(T1_list,FAs,TR, return_tensor): #K seems like it'd be the number of flip angles or time points
    K = FAs.size
    D = np.zeros((K, T1_list.shape[0])) #changed N_T1T2 to N_T1*N_T2 because the first doesn't make sense to me
    N_T1 = T1_list.shape[0]
    #N_T2 = N_T1
    tic = time.time()
    par_jobs = 5
    res = joblib.Parallel(n_jobs=par_jobs)(joblib.delayed(myfun_full)(x,FAs,TR) for x in T1_list)
    toc = time.time()
    print('Done creating dictionary:', toc - tic, 'seconds')
    D = np.array([r[0] for r in res])
    sc_list = np.array([r[1] for r in res])
    if return_tensor:
    	print("returning torch tensors...")
    	D = torch.from_numpy(D)
    	sc_list = torch.from_numpy(sc_list)
    return D, sc_list #output D to have the correct shape

def make_T1B1_dict(T1B1_list, FAs,TR, return_tensor):#make_T1B1_dict(T1_list,B1_list, FAs,TR, return_tensor): #K seems like it'd be the number of flip angles or time points
    K = FAs.size
    N = T1B1_list.shape[1]
    D = np.zeros((K, N)) #changed N_T1T2 to N_T1*N_T2 because the first doesn't make sense to me

    tic = time.time()
    par_jobs = 5
    res = joblib.Parallel(n_jobs=par_jobs)(joblib.delayed(myfun_full_b1corr)(T1B1_list[:,ii], TR, FAs) for ii in range(0, N))#for ii in T1_list for jj in B1_list)
    toc = time.time()
    print('Done creating dictionary:', toc - tic, 'seconds')
    D = np.array([r[0] for r in res])
    sc_list = np.array([r[1] for r in res])
    if return_tensor:
        print("returning torch tensors...")
        D = torch.from_numpy(D)
        sc_list = torch.from_numpy(sc_list)
    return D, sc_list #output D to have the correct shape


#function that matches the image SI to the dictionary key
def dictionary_match(im_data, D, t1_list, sc_list):
    # im_data shape is [nt, nx, ny]
    nx = im_data.shape[1]
    ny = im_data.shape[2]
    nt = im_data.shape[0]
    im_data_rs = torch.transpose(torch.reshape(im_data,(nt,nx*ny)),0,1) #no longer need to take abs() since input is magnitude image
    im_data_rs = im_data_rs[...,None]
    #print("im_data_rs.shape = ", im_data_rs.shape)
    
    # reshape D to be (1, K, N_T1)
    #D_sc = torch.transpose(torch.mul(torch.transpose(D,0,1),sc_list),0,1) #make a copy of D that's rescaled!
    D_rs = torch.transpose(D,0,1) # replace np with torch.blah 
    D_rs = D_rs[None,...]
    #lines up to line 48 don't change over time, so this is also unecessary to do every iteration 
    #print("D_rs.shape = ", D_rs.shape)

    # dictionary matching
    #z = np.sum(np.dot(im_data_rs, D_rs),dim=1)
#    z = torch.sum(im_data_rs*D_rs, axis=1) #same so far...
#    print("z.shape = ", z.shape) # same
#    i = torch.argmax(z,axis=1) #same yay
    #print("index.shape = ", i.shape)
    # dictionary matching that we want to do in batches

    if im_data_rs.shape[0] >20000:
        batch_width = 1
        batch_num = np.int(D_rs.shape[2]/batch_width)
        #print("batch_num = ",batch_num)
        z = torch.zeros((im_data_rs.shape[0],D_rs.shape[2])).to(D_rs.device)
        i_idx = torch.zeros((im_data_rs.shape[0]),dtype=torch.long)
        for i in range(1,batch_num+1):
            z_temp = torch.sum(im_data_rs*D_rs[:,:,((i-1)*batch_width):(batch_width*(i))], axis=1)
            z[:,((i-1)*batch_width):(batch_width*(i))] = z_temp

            #display(z_temp.shape)
        batch_width2 = 2
        while z.shape[0] % batch_width2 != 0:
            batch_width2 = batch_width2+1
        batch_num2 = np.int(z.shape[0]/batch_width2)
        for j in range(1,batch_num2+1):
            i_temp = torch.argmax(z[((j-1)*2):(2*(j)),:],axis=1)
            i_idx[((j-1)*2):(2*(j))] = i_temp
    else:
        #z = np.sum(np.dot(im_data_rs, D_rs),dim=1)
        z = torch.sum(im_data_rs*D_rs, axis=1) #same so far...
        #print("z.shape = ", z.shape) # same
        i_idx = torch.argmax(z,axis=1) #z -> abs(z) for this argmax calculation
    #print("index.shape = ", i_idx.shape)
    z_idx = z[...,i_idx]
    #print(z_idx.shape)
    # get scaling
    sc = torch.transpose(z_idx,0,1) # this would now be complex valued
    #print("sc.shape = ", sc.shape)
    scales = torch.max(sc,axis=0)[0] #this would change to abs()
    #idx = torch.argmax(abs(blah))
    #scales_final = sc[idx]
    
    #print(scales.shape)
    model_out = D[i_idx,:]*scales[...,None] #model_out is D indexed accoringly
    #print("model_out.shape = ",model_out.shape)
    model_out_im = torch.reshape(model_out,(nx,ny,nt)).permute(2,0,1) #yep
    #print("model_out_im.shape = ", model_out_im.shape)
    return scales, i_idx, model_out_im



#the model, in this case the SPGRE equation that takes in [T1,T2, image_params] and outputs SI
def my_sim(x,FAs, TR): #img_params = {[FAs],TR,TE}
    print("entered my_sim function...")
    f = lambda x,fas,tr: (np.sin(fas*np.pi/180)*(1-np.exp(-tr/x)))/(1-np.cos(fas*np.pi/180)*np.exp(-tr/x))
    z = f(x,FAs,TR) #maybe utils.complex.r2c?
    print("successfully evaluated f(x,FAs,TR)")
    zs = np.linalg.norm(z)
    #print(z.shape)
    #print(zs)
    return z / zs, zs

def myfun_full(t1, flips, tr):
    E1 = np.exp(-tr/t1)
    sina = np.sin(np.pi / 180 *flips)
    cosa = np.cos(np.pi / 180 *flips)
    z = sina * (1 - E1) / (1 - cosa * E1)
    zs = np.linalg.norm(z)
    return z / zs, zs

def myfun_full_b1corr(t1b1,tr, flips):
    t1 = t1b1[0]
    b1 = t1b1[1]
    E1 = np.exp(-tr/t1)
    sina = np.sin(flips*b1)
    cosa = np.cos(flips*b1)
    z = sina * (1 - E1) / (1 - cosa * E1)
    zs = np.linalg.norm(z)
    return z / zs, zs

make_t1_dict_now = False
make_b1_dict_now = False
if make_t1_dict_now:
    FAs = np.array([4,6,8, 10, 12, 14, 16, 18, 20])
    TR = 6 #6.1
    TE = 2.328 #2.75
    T1_min = 50
    T1_max = 4000 #ms
    N_T1 = 2000
    return_tensor = True
    #im_file = "/Users/Kalina/Desktop/x_gt_MDA_MFA.npy"
    #im_data = np.load("x_gt_MDA_MFA.npy")
    #nt = im_data.shape[0]
    #nx = im_data.shape[1]
    #ny = im_data.shape[2]

    print("file doesn't exist")
    t1_list = make_T1_list(T1_min,T1_max,N_T1)
    D,sc_list = make_T1_dict(t1_list,FAs,TR,return_tensor) #D is the dictionary of SI vs FA for N_T1 values. sc_list is the scale factor for N_t1 values
    if return_tensor:
        print("t1_list.shape", t1_list.shape)
        print("D.shape = ", D.shape)
        print("sc_list.shape = ", sc_list.shape)
        torch.save(D, "/home/kalina/Code/dictionaries/dict_T1mapping_TR_6_FAs_4-20_2k.pt")
        torch.save(sc_list, "/home/kalina/Code/dictionaries/sc_T1mapping_TR_6_FAs_4-20_2k.pt")
        torch.save(torch.from_numpy(t1_list), "/home/kalina/Code/dictionaries/t1_list_50_4000_2k.pt")
    else:
        np.save("dict_T1mapping_FAs_3_5_10_15",D)
        np.save("sc_T1mapping_FAs_3_5_10_15",sc_list)
        np.save("t1_list_0.01_3000_10k",t1_list)

if make_b1_dict_now:
    FAs = np.array([2,4,6,8, 10, 12, 14, 16, 18, 20])
    TR = 6
    TE = 2.328
    T1_min = 50
    T1_max = 4000 #ms
    B1_min = 0.7
    B1_max = 1.3
    N_B1 = 10
    N_T1 = 2000
    return_tensor = True
    #im_file = "/Users/Kalina/Desktop/x_gt_MDA_MFA.npy"
    #im_data = np.load("x_gt_MDA_MFA.npy")
    #nt = im_data.shape[0]
    #nx = im_data.shape[1]
    #ny = im_data.shape[2]

    print("file doesn't exist")
    t1_list = make_T1_list(T1_min,T1_max,N_T1)
    b1_list = make_B1_list(B1_min,B1_max,N_B1)
    t1b1_list = make_T1B1_list(t1_list,b1_list)
    
    D_b1,sc_list_b1 = make_T1B1_dict(t1b1_list,FAs,TR,return_tensor)
    if return_tensor:
        torch.save(D_b1, "/home/kalina/Code/deepinpy_private/modelling/dict_T1B1mapping_TR_6_FAs_2-20_20k.pt")
        torch.save(sc_list_b1, "/home/kalina/Code/deepinpy_private/modelling/sc_T1B1mapping_TR_6_FAs_2-20_20k.pt")
        torch.save(torch.from_numpy(t1b1_list), "/home/kalina/Code/deepinpy_private/modelling/t1b1_list_20k.pt")
    else:
        np.save("dict_T1B1mapping_FAs_2-20",D_b1)
        np.save("sc_T1B1mapping_FAs_2-20",sc_list_b1)
        np.save("t1b1_list_20k",t1b1_list)

