#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ismail Mert Meral, Yun Cheng, Xioxi He.
Semester Project: Deep Sensor Calibration Method with Multi-level sequence modelling
Computer Engineering and Networks Laboratory, TEC, Computer ETH Zürich
For support you can contact:
    mmeral@ethz.ch
    mertmeral@gmail.com
    
@author: mertmeral
"""

import os
import pandas as pd
import torch
from pickle import load

from main_new import evaluate_model_monthly
from utils_new import plotSequence

import argparse
import warnings; warnings.simplefilter('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Monthly evaluation of models SED and SEDglobal variants')

    parser.add_argument('--location', type=str, default='changping',
                            help='The name of the location')
    
    parser.add_argument('--modelType', type=str, default='NetSED',
                            help='Type of the model: NetSED, NetglobalSEDv1, NetglobalSEDv2, NetglobalSEDv3')
    
    parser.add_argument('--inputLength', type=int, default=24,
                            help='The length of the input sequence to the encoder.')
    
    parser.add_argument('--outputLength', type=int, default=1,
                            help='The length of the output sequence of the decoder.')
    
    parser.add_argument('--globalLength', type=int, default=8,
                            help='The number of weeks to extract global features from, only used for Netglobal*SED models')
    
    parser.add_argument('--globalFeatureType', type=str, default='pm25_only',
                            help='The type of the global features. Allowed types: {pm25_only, ts_features, all_channels_raw}.')
    
                        # pm25_only       : Global features consist of only raw pm2.5 values of the low cost sensor.
                        # ts_features     : Global features consist of weekly max,min,median and std of all measurement channels.
                        # all_channels_raw: Global features consist of raw values of all measurement channels.
                        
    parser.add_argument('--loss', type=str, default="mse",
                            help='The type of the loss. Allowed losses: mae, mse or dilate')

    parser.add_argument('--hiddenSize', type=int, default=1024,
                            help='Number of the GRU hidden states, same for Decoder and Encoder GRU.')
    
    parser.add_argument('--numLayers', type=int, default=1,
                            help='Number of GRU layers.')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                            help='The dropout rate. Must be in [0,1) range.')
    
    parser.add_argument('--epochs', type=int, default=3,
                            help='The number of training epochs.')
    
    parser.add_argument('--lr', type=float, default= 0.0001,
                            help='Initial learning rate for Adam optimizer.')

    parser.add_argument('--batchSize', type=int, default=32,
                            help='Batch size.')
    
    args = parser.parse_args()
    
    

    args = parser.parse_args()
    
    file_names = [s for s in os.listdir("data/Beijing/co-located") if s.endswith('.csv')]
    ds = {}
    for file in file_names:
        ds[file.replace('.csv', '')] = pd.read_csv("data/Beijing/co-located/"+file)
        
    ds_name= args.location;
    s1_df = ds[ds_name+"_compare"]
    #Linealy interpolate missing values
    s1_df = s1_df.interpolate(method ='linear', limit_direction ='forward')
    
    
    input_length        = args.inputLength
    output_length       = args.outputLength
    batch_size          = args.batchSize
    global_length       = args.globalLength
    global_feature_type = args.globalFeatureType
    
    model_parent_dir = 'saved_models/'+ds_name+'/'+args.modelType+'_inputLength'+str(args.inputLength)+'_outputLength'+str(args.outputLength)+'_batchSize'+str(args.batchSize)+'_epochs'+str(args.epochs)

    model_path             = os.path.join(model_parent_dir+'/net_.pt')
    scalerX_filename       = os.path.join(model_parent_dir+'/scalerX_.pkl')
    scalerY_filename       = os.path.join(model_parent_dir+'/scalerY_.pkl')
    scalerXglobal_filename = os.path.join(model_parent_dir+'/globalScaler_.pkl')


    with open(model_path, 'rb') as f:
        net = torch.load(f)
    with open(scalerX_filename, 'rb') as f2:
        scalerX = load(f2)
    with open(scalerY_filename, 'rb') as f1:
        scalerY = load(f1)
    with open(scalerXglobal_filename, 'rb') as f3:
        scalerX_global = load(f3)
        
    labels,predicts = evaluate_model_monthly(s1_df, net,scalerX, scalerY, scalerX_global, input_length, output_length,global_length, global_feature_type)  
    l,p,means,stds = plotSequence(labels,predicts)
    
    

