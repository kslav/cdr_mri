#code for making the dictionary
import numpy as np
import time
import joblib
import sys
#sys.path.append("/Users/kalina/Downloads/deepinpy_private_201208/")
import deepinpy.utils.complex as cp
import torch
import scipy.io as scp
#import module utils

def make_PK_list(ktrans_min,ktrans_max,ve_min, ve_max, N_ktrans, N_ve):
    ktrans_list = np.linspace(ktrans_min, ktrans_max, N_ktrans)
    ve_list = np.linspace(ktrans_min, ktrans_max, N_ktrans)
    print("ktrans_list.shape = ", ktrans_list.shape)
    print("ve_list.shape = ", ve_list.shape)
    return ktrans_list[...,None], ve_list[...,None] # np.dstack(ktrans_list, ve_list)

def make_R1_list(R1_min,R1_max,N_R1):
    T1_list = np.linspace(R1_min, R1_max, N_R1)
    print(T1_list.shape)
    return T1_list[...,None]

def make_PK_dict(ktrans_list,ve_list, t,Cp,FA,TR,r1_list,Rel, return_tensor): #K seems like it'd be the number of flip angles or time points
    K = Cp.size
    D = np.zeros((K, ktrans_list.shape[0])) #changed N_T1T2 to N_T1*N_T2 because the first doesn't make sense to me
    N_ktrans = ktrans_list.shape[0]

    tic = time.time()
    par_jobs = 5
    res = joblib.Parallel(n_jobs=par_jobs)(joblib.delayed(myPatlakFun_SI)(x,t,Cp,FA,TR,y,Rel) for x in ktrans_list for y in r1_list)
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
    #print("D_rs.shape = ", D_rs.shape)

    # dictionary matching
    #z = np.sum(np.dot(im_data_rs, D_rs),dim=1)
    z = torch.sum(im_data_rs*D_rs, axis=1) #same so far...
    #print("z.shape = ", z.shape) # same
    i = torch.argmax(z,axis=1) #same yay
    #print("index.shape = ", i.shape)
    
    # get scaling
    sc = torch.transpose(z[...,i],0,1) #/ sc_list[i] #changing from z[i] because tiny fake data produces a bug...
    #print("sc.shape = ", sc.shape)
    scales = torch.max(sc,axis=0)[0]
    #print(scales.shape)
    model_out = D[i,:]*scales[...,None] #model_out is D indexed accoringly
    #print("model_out.shape = ",model_out.shape)
    model_out_im = torch.reshape(model_out,(nx,ny,nt)).permute(2,0,1) #yep
    x_np = im_data.cpu().numpy()
    x_reco_np = model_out_im.cpu().numpy()
    print("NRMSE_current = ", np.linalg.norm(abs(x_np) - x_reco_np) / np.linalg.norm(abs(x_np)))
    #print("model_out_im.shape = ", model_out_im.shape)
    return scales, i, model_out_im

def myfun_full_T1(t1, flips, tr):
    E1 = np.exp(-tr/t1)
    sina = np.sin(np.pi / 180 *flips)
    cosa = np.cos(np.pi / 180 *flips)
    z = sina * (1 - E1) / (1 - cosa * E1)
    zs = np.linalg.norm(z)
    return z / zs, zs

###########################################

def myToftsFun_SI(x,t,Cp,flips,TR,R10,Rel):
# Kety-Tofts equation for single voxel
# x = [Ktrans, ve pair]
    n_images = np.size(t)
    dcesin = np.sin(np.pi / 180 *flips)
    dcecos = np.cos(np.pi / 180 *flips)
    SI = np.zeros(n_images)
    SI[0] = dcesin*(1-np.exp(-TR*R10))/(1-dcecos*np.exp(-TR*R10)) # SPGE eqn.

    for k in range(1,n_images):
        int_t = t[k]
        for j in range(0,k):
            dummy_t = t[j]
            expo[j] =np.exp(-((x[1]/x[2]*(int_t-dummy_t))))
            crpexp[j] = Cp[j]*expo[j]
      
        t2 = t[0:k]
        crpexp_integral = np.trapz(crpexp, t2)
        
        c_toi[k] = x[1]*crpexp_integral
        R1[k]=(Rel*c_toi[k])+R10
        #     R1(k)=Rel*(1/0.8)*c_toi(k)+R1_vox; % Some people set fraction of
        #     water accessible to CA to 0.8, some people set it to 1. Anum set it
        #     to 0.8 for preclinical Vandy data analysis, but will set it to 1 for
        #     all future UT analysis (clinical and preclinical)
        
        SI[k]= dcesin*(1-np.exp(-TR*R1[k]))/(1-dcecos*np.exp(-TR*R1[k]))

    SI = np.double(SI)
    zs = np.linalg.norm(SI)
    return SI / zs, zs

############################################
def myPatlakFun_SI(x,t,Cp,flips,TR,R10,Rel):
# Patlak siimplification of Ket-Tofts for single voxel
    #print(np.size(Cp))
    n_images = t.size
    print("n_images = ", n_images)
    dcesin = np.sin(np.pi / 180 *flips)
    dcecos = np.cos(np.pi / 180 *flips)
    SI = np.zeros(n_images)
    R1 = np.zeros(n_images)
    c_toi = np.zeros(n_images)
    print(SI.shape)
    SI[0] = dcesin*(1-np.exp(-TR*R10))/(1-dcecos*np.exp(-TR*R10)) # SPGE eqn.

    for k in range(1,n_images):
        int_t = t[k]
        expo = np.zeros(k)
        crpexp = np.zeros(k)
        for j in range(0,k):

            dummy_t = t[j]
            dim = np.shape(int_t-dummy_t)
            expo[j] = np.ones(dim) #exp(-((x(1)/x(2).*(int_t-dummy_t))));
            crpexp[j] = Cp[j]*expo[j]
        t2 = t[0:k]
        crpexp_integral = np.trapz(crpexp,t2)
        
        c_toi[k] = x*crpexp_integral
        #size(x)
        R1[k]=(Rel*c_toi[k])+R10;
        #     R1(k)=Rel*(1/0.8)*c_toi(k)+R1_vox; % Some people set fraction of
        #     water accessible to CA to 0.8, some people set it to 1. Anum set it
        #     to 0.8 for preclinical Vandy data analysis, but will set it to 1 for
        #     all future UT analysis (clinical and preclinical)
        
        SI[k]= dcesin*(1-np.exp(-TR*R1[k]))/(1-dcecos*np.exp(-TR*R1[k]))

    SI = np.double(SI);
    zs = np.linalg.norm(SI)
    return SI / zs, zs

##########################################


make_dict_now = False
if make_dict_now:
    FA = 6
    TR = 7.598
    TE = 2.328
    R10 = 1/1500
    Rel = 5.5
    #Cp_TXO = np.array([0, 0, 0, 0, 0,0, 0, 2.06131362295522, 3.42251072802099, 4.50012241727796, 4.95369381456293, 4.83617882527277, 4.36729942697957, 3.76079521208048, 3.15931287320679, 2.63429485477876, 2.20819382213334, 1.87688025765588, 1.62522722102959, 1.43587949565351, 1.29328005236807, 1.18497929027923, 1.10162825612872, 1.03646549951258, 0.984702083436002, 0.942974513647964, 0.908915791347232, 0.880841921862787, 0.857532710825663, 0.838082908118561, 0.821803041757500, 0.808154085020965, 0.796704566724626, 0.787102289103818, 0.779055426968136, 0.772319603524316, 0.766688766697223, 0.761988496538816, 0.758070892794690, 0.754810518650424, 0.752101079279957, 0.749852637283114, 0.747989241143463, 0.746446886593335, 0.745171756331042, 0.744118698396673, 0.743249912240078, 0.742533816823185, 0.741944078562546, 0.741458779420346, 0.741059707467284, 0.740731754008116, 0.740462402989594, 0.740241299942229, 0.740059889150225, 0.739911109093549, 0.739789137454557, 0.739689178122783, 0.739607283662506, 0.739540207629199, 0.739485281936538, 0.739440315191275, 0.739403508536310, 0.739373386081092, 0.739348737461531, 0.739328570467466])
    Cp_TXO = scp.loadmat('/Users/Kalina/Desktop/AIF_pop_TXO')['AIF_pop']
    #print(Cp_TXO)
    t_AIF = scp.loadmat('/Users/Kalina/Desktop/t_AIF')['t_new']
    print("t.size = ",t_AIF.shape)
    print("Cp_TXO.size = ", Cp_TXO.shape)
    ktrans_min = 0.001
    ktrans_max = 4.9 #s^-1
    N_ktrans = 10000
    ve_min = 0.001
    ve_max = 1 #ms
    N_ve = 10000
    return_tensor = True
    #im_file = "/Users/Kalina/Desktop/x_gt_MDA_MFA.npy"
    #im_data = np.load("x_gt_MDA_MFA.npy")
    #nt = im_data.shape[0]
    #nx = im_data.shape[1]
    #ny = im_data.shape[2]

    print("file doesn't exist")
    ktrans_list, ve_list = make_PK_list(ktrans_min,ktrans_max,ve_min, ve_max, N_ktrans, N_ve)
    D_PK,sc_list_PK = make_PK_dict(ktrans_list,ve_list, t_AIF[0,7:],Cp_TXO[0,7:],FA,TR,R10,Rel, return_tensor) #D is the dictionary of SI vs FA for N_T1 values. sc_list is the scale factor for N_t1 values
    if return_tensor:
        #print("t1_list.shape", t1_list.shape)
        print("D_PK.shape = ", D_PK.shape)
        print("sc_list_PK.shape = ", sc_list_PK.shape)
        torch.save(D_PK, "/Users/Kalina/Desktop/dict_PKmapping_TEST.pt")
        torch.save(sc_list_PK, "sc_PKmapping_TEST.pt")
        #torch.save(torch.from_numpy(t1_list), "t1_list_1_4000_15k_v2.pt")
        torch.save(torch.from_numpy(ktrans_list), "/Users/Kalina/Desktop/ktrans_list_0p001_4p9_10k.pt")
        torch.save(torch.from_numpy(ve_list), "/Users/Kalina/Desktop/ve_list_0p001_1_10k.pt")
    else:
        np.save("dict_PKmapping_TEST",D_PK)
        np.save("sc_PKmapping_TEST",sc_list_PK)
        np.save("ktrans_list_0p001_4p9_10k", ktrans_list)
        np.save("ve_list_0p001_1_10k", ve_list)

