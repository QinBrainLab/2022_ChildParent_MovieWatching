
# -*- coding: utf-8 -*-
"""

@author: SU
Compute Pair-wise ISC among all subjects
"""
import nibabel as nib
from nilearn import masking 
import numpy as np
import os 
import pandas as pd
from os.path import join
from scipy.stats import  zscore
os.chdir('/home/suhaowen/CPVN_PY/isc-tutorial-master')
from isc_standalone import (isc, load_images,
                            load_boolean_mask, mask_images,
                            MaskedMultiSubjectData)



##read filelist
os.chdir('/home/suhaowen/CheckData_20200226/41subISC/list')
cpvnlist = pd.read_csv('41filelist.csv')
print(cpvnlist.shape)
mmm=r'/home/suhaowen/NewPipeline/template/Gray_matter_Template_2mm.nii'
for i in range(3321):#n--行数child 编号，i控制的是列数parent 编号
    Child=cpvnlist.iloc[i,1]
    Parent=cpvnlist.iloc[i,2]
    data_fmri_CP_dir='/home/suhaowen/CheckData_20200226/LMEcode'
    template_dir='/home/suhaowen/NewPipeline/template'
    
    swcaname='fmri/VN/smoothed_spm12/swcarI_6_hm_wm_csf_140s.nii.gz'
    child_path = join(data_fmri_CP_dir,Child,swcaname)
    parent_path =join(data_fmri_CP_dir,Parent,swcaname)
     
    print(child_path)
    print(parent_path)
# Filenames for MRI data; gzipped NIfTI images (.nii.gz)
    func_fns=[child_path,parent_path]
    mask_fn = join(template_dir, 'Gray_matter_Template_2mm.nii')
#    mni_fn = join(data_dir1, 'MNI152_T1_2mm_brain.nii.gz')

# Load a NIfTI of the brain mask as a reference Nifti1Image
    ref_nii = nib.load(mask_fn)
# Load functional images and masks using brainiak.io
    func_imgs = load_images(func_fns)
    mask_img = load_boolean_mask(mask_fn)
# Get coordinates of mask voxels in original image
    mask_coords = np.where(mask_img)
# Apply the brain mask using brainiak.image
    masked_imgs = mask_images(func_imgs, mask_img)
# Collate data into a single TR x voxel x subject array
    orig_data = MaskedMultiSubjectData.from_masked_images(masked_imgs,
                                                  len(func_fns))

    
   
# Z-score time series for each voxel
    orig_data = zscore(orig_data, axis=0)
#compute ISC 
    print(f"Original fMRI data shape: {orig_data.shape} "
      f"\ni.e., {orig_data.shape[0]} time points, {orig_data.shape[1]} voxels, "
      f"{orig_data.shape[2]} subjects")
    iscs=isc(orig_data, pairwise=True, summary_statistic=None, tolerate_nans=True)


    file_save_dir='/home/suhaowen/ISC/output'
    savename=cpvnlist.iloc[i,0]
    print(savename)
    #nilearn try
    iscs_nii=masking.unmask(iscs,mmm)   
    os.chdir(file_save_dir)
    nib.save(iscs_nii,savename)
    del iscs 

   