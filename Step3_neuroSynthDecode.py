# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:28:55 2020

@author: 92175
"""
from neurosynth import Dataset
from neurosynth import decode
import os
filepath=r'C:\Users\92175\Desktop\temp\reverseInference'
os.chdir(filepath)
dataset=Dataset('database.txt')

dataset.add_features('features.txt')



###---------------15TOPIC NC----------------#
decoder=decode.Decoder(dataset,features=['switching',
'anticipation',
'episodic',
'inhibition',
'music',
'social',
'face',
'auditory',
'sensorimotor',
'conflict',
'somatosensory',
'feedback',
'pain',
'emotion',
'language'])


decoder.decode([r'E:\41subReverse\fdr_0.05_CPs_CPo.nii'],save=r'E:\41subReverse\decoding_vmpfc_15topics.txt')