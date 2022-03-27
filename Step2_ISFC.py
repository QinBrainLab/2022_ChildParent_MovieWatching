

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:57:15 2019

@author: SU

"""
import nibabel as nib
from nilearn import masking 
import numpy as np
import os 
import pandas as pd
from os.path import join
from scipy.stats import  zscore
from brainiak.isc import isc
from nilearn.masking import apply_mask 
from nilearn.masking import unmask
from scipy.stats import pearsonr
import os
from nilearn import input_data
##循环读取文件
os.chdir('/home/suhaowen/CheckData_20200226/41subISC/list')
cpvnlist = pd.read_csv('41filelist.csv')

print(cpvnlist.shape)

SeedPeak=[(2,38,-18)]
# SeedPeak=[(2,38,-18)]
mmm=r'/home/suhaowen/NewPipeline/template/Gray_matter_Template_2mm.nii'
for i in range(0,3321):#n--行数child 编号，i控制的是列数parent 编号
    ISFC_signal1=[]
    ISFC_signal2=[]
    
    
    
    Child=cpvnlist.iloc[i,1]
    Parent=cpvnlist.iloc[i,2]
    data_fmri_CP_dir='/home/suhaowen/CheckData_20200226/LMEcode'
    data_dir1='/home/suhaowen/NewPipeline/template'
    
    swcaname='fmri/VN/smoothed_spm12/swcarI_6_hm_wm_csf.nii.gz'
    first_path = join(data_fmri_CP_dir,Child,swcaname)
    second_path =join(data_fmri_CP_dir,Parent,swcaname)
     
    print(first_path)
    print(second_path)
    
    print(first_path)
    print(second_path)
# Filenames for MRI data; gzipped NIfTI images (.nii.gz)
    func_fns=[first_path,second_path]
    mask_fn = join(data_dir1, 'Gray_matter_Template_2mm.nii')
#    mni_fn = join(data_dir1, 'MNI152_T1_2mm_brain.nii.gz')
   

    masker=input_data.NiftiSpheresMasker(SeedPeak, radius=6,detrend=False, standardize=False,
    low_pass=None, high_pass=None, t_r=2,memory='nilearn_cache', memory_level=6, verbose=2)
   
    SeedSignal1= masker.fit_transform(first_path,confounds=None)
   
    wholeBrainSignal1=apply_mask(second_path, mmm, dtype='f', smoothing_fwhm=None, ensure_finite=True)

    for j in range(154993):
        VoxelSignal1=wholeBrainSignal1[:,j]
        r_value1=pearsonr(SeedSignal1.flatten(),VoxelSignal1)[0]
        ISFC_signal1.append(r_value1)
    ISFC_final_signal1=np.array(ISFC_signal1)
    
    
    
    SeedSignal2= masker.fit_transform(second_path,confounds=None)
   
    wholeBrainSignal2=apply_mask(first_path, mmm, dtype='f', smoothing_fwhm=None, ensure_finite=True)

    for j in range(154993):
        VoxelSignal2=wholeBrainSignal2[:,j]
        r_value2=pearsonr(SeedSignal2.flatten(),VoxelSignal2)[0]
        ISFC_signal2.append(r_value2)
    ISFC_final_signal2=np.array(ISFC_signal2)
    

    
#    print len( np.nan_to_sum(ISFC_final_signal)
    z1=np.arctanh(ISFC_final_signal1.transpose())
    z2=np.arctanh(ISFC_final_signal2.transpose())
    
    ISFC_final_signal=(z1+z2)/2

    ISFC_image=unmask(ISFC_final_signal, mmm, order='F')


    file_save_dir='/home/suhaowen/isfc_output'
    savename=cpvnlist.iloc[i,2]
    print(savename)
#    iscs_nii=masking.unmask(iscs,mmm,order='F')   
    os.chdir(file_save_dir)
    nib.save(ISFC_image,savename)
   
#ff=os.listdir(r'/home/suhaowen/NewPipeline/ISFC/output/firstSeed')