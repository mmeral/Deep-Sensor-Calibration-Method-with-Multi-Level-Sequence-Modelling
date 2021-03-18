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
import numpy as np
import pandas as pd
import random
import argparse

import torch
from torch.utils.data import DataLoader


from utils_new import customDataset, getScalers, getGlobalScaler
from utils_new import split_dataset
from utils_new import monthly_train_val_split
from utils_new import ModelSaver
from utils_new import plotSequence

from model_new import EncoderRNN, DecoderRNN, Net_SED
from model_new import EncoderRNN_withGlobal, DecoderRNN_withGlobal, Net_global_SEDv1
from model_new import Net_global_SEDv2
from model_new import Net_global_SEDv3
from model_new import globalReducer,EncoderRNN_withGlobal_v2,DecoderRNN_withGlobal_v2
from model_new import Net_global_SEDv4
from model_new import EncoderRNN_withGlobal_v3,DecoderRNN_withGlobal_v3

from loss.dilate_loss import dilate_loss
from tslearn.metrics import dtw_path

import warnings; warnings.simplefilter('ignore')



def train_model(net, trainloader, validationloader, loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5, scalerY=[], modelSaver=[],save=True,device=[]):
    
    
 
    #Using Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()
    criterion2= torch.nn.L1Loss()
    
    train_loss_hist   = []  # save training loss at every epoch
    val_mse_hist      = []  # save validation *mse* at every epoch such that epoch%print_every == 0
    val_mae_hist      = []  # save validation *mae* at every epoch such that epoch%print_every == 0
    val_dtw_hist      = []  # save validation *dtw* at every epoch such that epoch%print_every == 0, for DILATE Loss
    val_tdi_hist      = []  # save validation *tdi* at every epoch such that epoch%print_every == 0, for DILATE Loss
    
    #scale: constant multiplier of the output normalizer scalerY.
    #       An output value X(in the original range) is scaled as X*scale(to the zero-max normalized range) 
    
    scale = torch.from_numpy(scalerY.scale_).to(device).float()
    
    
    train_loss_epoch = [] #temporary storage for losses over batches in a single epoch
    best_loss = 10000000000000  #track the best VALIDATION loss for early stopping
    
    
    for epoch in range(epochs):
        net.train()
        train_loss_epoch = []
        
        # You can schedule the learning rate by modifying here,
        # E.g. At 10th and 25th epoch, divide the learning rate by 10 as:
        # if(epoch==10 or epoch==25):
        #     for param_group in optimizer.param_groups:
        #         learning_rate = learning_rate*0.1
        #         param_group['lr'] = learning_rate
        
        
        for i, data in enumerate(trainloader, 0):
            
            
            #Get a batch of data from the trainloader
            inputs_Encoder, inputs_Decoder, inputs_Encoder_global, inputs_Decoder_global, target = data
            
            inputs_Encoder = torch.tensor(inputs_Encoder, dtype=torch.float32).to(device)
            inputs_Decoder = torch.tensor(inputs_Decoder, dtype=torch.float32).to(device)
            inputs_Encoder_global = torch.tensor(inputs_Encoder_global, dtype=torch.float32).to(device)
            inputs_Decoder_global = torch.tensor(inputs_Decoder_global, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]

            # Forward  
            outputs = net(inputs_Encoder, inputs_Decoder, inputs_Encoder_global, inputs_Decoder_global)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            # Loss
            if (loss_type=='mse'):
                
                loss_mse = criterion(target[:,:,0]*scale,outputs[:,:,0])
                loss = loss_mse
 
            if (loss_type=='dilate'):
                
                loss, loss_shape, loss_temporal = dilate_loss(target*scale,outputs,alpha, gamma, device)
                
            if(loss_type=='mae'):
                
                loss_mae = criterion2(target[:,:,0]*scale,outputs[:,:,0])
                loss = loss_mae
              
            # Backward + Optimize    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
            
        train_loss_hist.append(np.mean(train_loss_epoch))
        
   
        if(verbose):
            if (epoch % print_every == 0):
                #Evaluate on validation       
                net.eval()
                val_mse, val_mae, val_dtw, val_tdi = eval_model(net,validationloader,device,gamma =gamma,verbose=1,scalerY=scalerY)
                print('Epoch: ', epoch, '| Train loss: ',np.mean(train_loss_epoch), '| Val mae loss: ', val_mae*scalerY.scale_)
                print('|--------------------------------------------------------|')
                val_mse_hist.append(val_mse)
                val_mae_hist.append(val_mae)
                val_dtw_hist.append(val_dtw)
                val_tdi_hist.append(val_tdi)
                
                
                if(val_mae<best_loss):
                    #save model with respect to validation mae
                    best_loss = val_mae
                    if(save):
                        modelSaver.saveNetwork(network)
                        modelSaver.best_epoch = epoch
                        modelSaver.best_net = network


    
    return train_loss_hist, val_mse_hist, val_mae_hist, val_dtw_hist, val_tdi_hist

def eval_model(net,loader,device, gamma,verbose=1,scalerY=[]):
    

    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    
    losses_mse = []
    losses_dtw = []
    losses_tdi = []
    losses_mae = []
    losses_mae_lastPoint = []
    losses_linearWeighted = []
    
    # scale: constant multiplier of the output normalizer scalerY.
    scale = torch.from_numpy(scalerY.scale_).to(device).float()
    
    # w: Weights for output sequence for weighted evaluation, linearly increasing towards the latest time point.
    output_size = net.target_length
    w = np.arange(output_size) + 1
    w = w * output_size / np.sum(w)
    w = torch.from_numpy(w).to(device)

    
    for i, data in enumerate(loader, 0):
        
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        
        inputs_Encoder, inputs_Decoder, inputs_Encoder_global, inputs_Decoder_global, target = data
        
        inputs_Encoder = torch.tensor(inputs_Encoder, dtype=torch.float32).to(device)
        inputs_Decoder = torch.tensor(inputs_Decoder, dtype=torch.float32).to(device)
        inputs_Encoder_global = torch.tensor(inputs_Encoder_global, dtype=torch.float32).to(device)
        inputs_Decoder_global = torch.tensor(inputs_Decoder_global, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        
        #Predict
        outputs = net(inputs_Encoder, inputs_Decoder, inputs_Encoder_global, inputs_Decoder_global)

        #Evaluate the predicted sequences. Metrics:
        # 1: Mean squared error
        # 2: Mean absolute error averaged over the output sequence
        # 3: Mean absolute error of the last point in the output sequence
        # 4: Mean absolute error weighted averaged over the output sequence for the weights w
        
        loss_mse = criterion(torch.squeeze(target,-1)*scale,torch.squeeze(outputs,-1))
        loss_mae = criterion2(torch.squeeze(target,-1),torch.squeeze(outputs/scale,-1))
        loss_mae_lastPoint = criterion2(target[:,-1,0], outputs[:,-1,0]/scale)
        loss_mae_linearWeighted =  criterion2(target[:,:,0]*w, outputs[:,:,0]*w/scale)

 
        # If DTW and TDI is a metric of interest, you can calculate them by uncommenting here:
            
        #-----------------------------------------------------------------------------#    
        # loss_dtw, loss_tdi = 0,0
        # for k in range(batch_size):
        #     target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
        #     output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()
        #
        #     path, sim = dtw_path(target_k_cpu, output_k_cpu/scale.detach().cpu())
        #     loss_dtw += sim
        #               
        #     Dist = 0
        #     for i,j in path:
        #             Dist += (i-j)*(i-j)
        #     loss_tdi += Dist / (N_output*N_output)
        #                
        # loss_dtw = loss_dtw /batch_size
        # loss_tdi = loss_tdi / batch_size
        # losses_dtw.append( loss_dtw )
        # losses_tdi.append( loss_tdi ) 
        #-----------------------------------------------------------------------------#  
        
        

        losses_mse.append( loss_mse.item() )
        losses_mae.append( loss_mae.item() )
        losses_mae_lastPoint.append(loss_mae_lastPoint.item() )
        losses_linearWeighted.append(loss_mae_linearWeighted.item())
        

    print( 'Val MSE=',np.array(losses_mse).mean(),' Val MAEavg= ', np.array(losses_mae).mean())
    print( 'Val MAElast= ',np.array(losses_mae_lastPoint).mean(),' ValMAEweighted= ', np.array(losses_linearWeighted).mean())
    
    return np.array(losses_mse).mean(), np.array(losses_mae).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean()

def predict(net,loader,device,scalerY):

    # Preserves the order of the samples in the loader.
    
    
    # Put the network into evaluation mode (Otherwise the batchnorm parameters are changed!)
    net.eval()
    
    # scale: constant multiplier of the output normalizer scalerY.
    scale = torch.from_numpy(scalerY.scale_).to(device).float()
      
    for i, data in enumerate(loader, 0):

        inputs_Encoder, inputs_Decoder, inputs_Encoder_global, inputs_Decoder_global, target = data
        inputs_Encoder = torch.tensor(inputs_Encoder, dtype=torch.float32).to(device)
        inputs_Decoder = torch.tensor(inputs_Decoder, dtype=torch.float32).to(device)
        inputs_Encoder_global = torch.tensor(inputs_Encoder_global, dtype=torch.float32).to(device)
        inputs_Decoder_global = torch.tensor(inputs_Decoder_global, dtype=torch.float32).to(device)
        
        outputs = net(inputs_Encoder, inputs_Decoder, inputs_Encoder_global, inputs_Decoder_global)
        outputs = outputs/scale
        
        if(i==0):
            labels_all = target.cpu().detach().numpy()
            predicts_all = outputs.cpu().detach().numpy()
        else:
            labels_all = np.vstack([labels_all,target.cpu().detach().numpy()])
            predicts_all = np.vstack([predicts_all,outputs.cpu().detach().numpy()])

        
    return labels_all,predicts_all

def evaluate_model_monthly(s_df, net,scalerX, scalerY, scalerX_global, input_length, output_length, global_length, global_feature_type):
    
    '''
    Calculate and displays the mean absolute error of the model calibrations for each month in the dataset.

    :param pandas.DataFrame s_df: The dataframe containing the measurements
                                   -The last column must be the ground truth
                                   -The 5th column must be the low-cost-sensor data, not including the recording time

    :param torch.net net        : The calibration model
    
    :param sklearn.preprocessing.MinMaxScaler scalerX       : 0-max normalization model for input
    :param sklearn.preprocessing.MinMaxScaler scalerY       : 0-max normalization model for output
    :param sklearn.preprocessing.MinMaxScaler scalerX_global: 0-max normalization model for global features 
        
    :param int input_length   : The length of the input sequence
    :param int output_length  : The length of the output sequence
    :param int global_length  : The length of the global history, *in weeks*    
    :param str global_feature_type    : The type of the extracted global features. 
                                        Allowed values:
                                            pm25_only       : Global features consist of only raw pm2.5 values of the low cost sensor.
                                            ts_features     : Global features consist of weekly max,min,median and std of all measurement channels.
                                            all_channels_raw: Global features consist of raw values of all measurement channels.
   

    :return: np.ndarray labels  : Ground truth output *sequences*
    :return: np.ndarray predicts: Calibrated output *sequences*

    '''
    
    print("Starting monthly evaluation...")

    
    # Split all of the data, including training months
    X_Enc, X_Dec, y, X_global, Xg_Enc, Xg_Dec,_ = split_dataset(s_df,input_length,output_length,global_length,global_feature_type)

    
    print("Data read...")
        
    #normalize inputs
    for i in range(X_Enc.shape[0]):
        X_Enc[i,:,:] = scalerX.transform(X_Enc[i,:,:])
    for i in range(X_Dec.shape[0]):
        X_Dec[i,:,:] = scalerX.transform(X_Dec[i,:,:])
    for i in range(Xg_Enc.shape[0]):
        Xg_Enc[i,:,:] = scalerX_global.transform(Xg_Enc[i,:,:])
    for i in range(Xg_Dec.shape[0]):
        Xg_Dec[i,:,:] = scalerX_global.transform(Xg_Dec[i,:,:])
    
    print("Input normalized...")
        
    dataset = customDataset(X_Enc, X_Dec, Xg_Enc, Xg_Dec, y)
    loader = DataLoader(dataset, batch_size=net.encoder.batch_size,shuffle=False, num_workers=1,drop_last=True)

    labels,predicts = predict(net,loader,net.device,scalerY)
        
    months = np.unique( [s[:-11] for s in s_df.iloc[:,:].values[:,0]])

    max_input_length = input_length+ global_length*168

    df_cut = s_df.iloc[max_input_length:]
    df_cut = df_cut[0: - (df_cut.shape[0] % net.encoder.batch_size)]
    
    for month in months:
        indicies  = df_cut.index[df_cut.time.str.contains(month,case=False)]-max_input_length
        
        if(indicies.shape[0] != 0):
            month_labels   = labels[indicies]
            month_predicts = predicts[indicies]
            print(month,', MAE:', np.mean(np.abs(month_labels-month_predicts)))
        else:
            print(month,', MAE: no data available due to the input length')
    return labels,predicts


            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training settings for the calibration models SED and SEDglobal variants')


    parser.add_argument('--location', type=str, default='changping',
                            help='The name of the location')
    
    parser.add_argument('--modelType', type=str, default='NetSED',
                            help='Type of the model: NetSED, NetglobalSEDv1, NetglobalSEDv2, NetglobalSEDv3,NetglobalSEDv4, NetglobalSEDv5')
    
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
    
    
    
    last_train_month = '2017-11'  #The last month of the training data
    last_val_month   = '2018-01'  #The last month of the validation data
    # Training period : from the first month to last_train_month
    # Validation period : from the last_train_month+1month to last_val_month

    
    #Read the name of the all locations. The data for location "myLocation" must be saved under ./data/Beijing/co-located/myLocation_compare.csv
    file_names = [s for s in os.listdir("data/Beijing/co-located") if s.endswith('.csv')]
    ds = {}
    for file in file_names:
        ds[file.replace('.csv', '')] = pd.read_csv("data/Beijing/co-located/"+file)
        
    ds_name= args.location
    s1_df = ds[ds_name+"_compare"]
    
    #Linearly interpolate missing values
    s1_df = s1_df.interpolate(method ='linear', limit_direction ='forward')
    
    #Set up the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    
    input_length = args.inputLength
    output_length = args.outputLength
    batch_size = args.batchSize
    global_feature_length = args.globalLength
    
    
    # global_feature_length_max: The maximum number of weeks allowed as global history. Fixed to 8. 
    # args.globalLength must be smaller than global_feature_length_max
    
    global_feature_length_max = 8 #max length
    global_length = global_feature_length*168  #convert weeks to days
    global_feature_type = args.globalFeatureType;

    # Splits the data with a sliding window and extracts encoder inputs, decoder inputs, ground truth sequences

    X_Enc, X_Dec, y, X_global, Xg_Enc, Xg_Dec,time_idx = split_dataset(s1_df,input_length,output_length,global_feature_length_max,global_feature_type)
      
    
    # Training-validation split with respect to the borders set with "last_train_month" and "last_val_month"
    # This function does not return the test set. The remaning data after last_val_month is not used.
    # The evaluation on the test months can be inspected with eval_model_monthly function, which returns the monthly errors for all months including the test months.
    
    X_Enc_train, X_Enc_val,Xg_Enc_train, Xg_Enc_val, y_train, y_val,tr_idx, val_idx = monthly_train_val_split(X_Enc,Xg_Enc,y,time_idx,last_train_month,last_val_month,args.inputLength,args.outputLength)
    X_Dec_train ,X_Dec_val,Xg_Dec_train, Xg_Dec_val, _,_,_,_                        = monthly_train_val_split(X_Dec,Xg_Dec,y,time_idx,last_train_month,last_val_month,args.inputLength,args.outputLength)


    #  Split_dataset extracted global features with the maximum allowed global_feature_length_max value for train set consistency,
    #  however, the model needs last global_length number of features out of the extracted global_feature_length_max*168 global features.
    
    Xg_Enc_train = Xg_Enc_train[:,:,-global_length:]
    Xg_Dec_train = Xg_Dec_train[:,:,-global_length:]
    Xg_Enc_val   = Xg_Enc_val[:,:,-global_length:]
    Xg_Dec_val   = Xg_Dec_val[:,:,-global_length:]
    X_global     = X_global[:,-global_length:]
    
    global_feature_size = Xg_Enc_train.shape[2]
        
    # Training scalers only on the training samples, indexed by tr_idx
    # scalerX: applies 0-max normalization to the GRU inputs (i.e. short history)
    # scalerY: applies 0-max normalization to the output
    # scalerX_global: applies 0-max normalization to the globalReducer inputs (i.e. global history)
    
    scalerX, scalerY = getScalers(s1_df, [x+global_feature_length_max*168 for x in tr_idx])
    scalerX_global   = getGlobalScaler(X_global,tr_idx)
    
    # Normalize encoder GRU inputs
    for i in range(X_Enc_train.shape[0]):
        X_Enc_train[i,:,:]  = scalerX.transform(X_Enc_train[i,:,:])
    for i in range(X_Enc_val.shape[0]):
        X_Enc_val[i,:,:]    = scalerX.transform(X_Enc_val[i,:,:])
    for i in range(Xg_Enc_train.shape[0]):
        Xg_Enc_train[i,:,:] = scalerX_global.transform(Xg_Enc_train[i,:,:])
    for i in range(Xg_Enc_val.shape[0]):
        Xg_Enc_val[i,:,:] = scalerX_global.transform(Xg_Enc_val[i,:,:])
        
    # Normalize decoder GRU inputs
    for i in range(X_Dec_train.shape[0]):
        X_Dec_train[i,:,:] = scalerX.transform(X_Dec_train[i,:,:])
    for i in range(X_Dec_val.shape[0]):
        X_Dec_val[i,:,:] = scalerX.transform(X_Dec_val[i,:,:])
    for i in range(Xg_Dec_train.shape[0]):
        Xg_Dec_train[i,:,:] = scalerX_global.transform(Xg_Dec_train[i,:,:])
    for i in range(Xg_Dec_val.shape[0]):
        Xg_Dec_val[i,:,:] = scalerX_global.transform(Xg_Dec_val[i,:,:])
        
        
        
    dataset_train      = customDataset(X_Enc_train, X_Dec_train, Xg_Enc_train, Xg_Dec_train, y_train)
    dataset_validation = customDataset(X_Enc_val  , X_Dec_val,   Xg_Enc_val,   Xg_Dec_val,   y_val  )

  

    # Drop_last=True, otherwise if (number_of_samples)mod(batchsize) != 0, error in the last batch. Be aware that last few inputs may be discarded due to this.
    
    trainloader      = DataLoader(dataset_train,      batch_size=batch_size,shuffle=True , num_workers=1,drop_last=True)
    validationloader = DataLoader(dataset_validation, batch_size=batch_size,shuffle=False, num_workers=1,drop_last=True)
    
    dataset_train      = []
    dataset_validation = []
    
    networkType = args.modelType

    # Initialize the encoder, decoder (globalReducer if applicable) and the network 
    # NetglobalSEDv4,v5 and v6 are developed differently. You select what you want to process your global features with
    # by selecting globalReducer.
    
    if networkType == 'NetSED':
        # No global features, default encoder decoder architecture
        encoder = EncoderRNN(input_size=14, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers,  batch_size=batch_size, dropout = args.dropout).to(device)
        decoder = DecoderRNN(input_size=15, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers,  output_size=1, batch_size=batch_size, dropout=args.dropout).to(device)
        network = Net_SED(encoder,decoder, output_length, args.dropout, device).to(device)
        
    elif networkType == 'NetglobalSEDv1':
        
        # Raw global features are appended to the current GRU hidden state.
        # calibrated output = FC([current GRU hidden state,raw global features])
        
        encoder = EncoderRNN_withGlobal(input_size=14, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size).to(device)
        decoder = DecoderRNN_withGlobal(input_size=15, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, output_size=1, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size).to(device)
        network = Net_global_SEDv1(encoder,decoder, output_length, args.dropout, device).to(device)
   
    elif networkType == 'NetglobalSEDv2':
        
        # Raw global features are appended to the GRU inputs.
        
        encoder = EncoderRNN(input_size=14+global_feature_size, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers,  batch_size=batch_size, dropout = args.dropout).to(device)
        decoder = DecoderRNN(input_size=15+global_feature_size, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers,  output_size=1, batch_size=batch_size, dropout=args.dropout).to(device)
        network = Net_global_SEDv2(encoder,decoder, output_length, args.dropout, device).to(device)

    elif networkType == 'NetglobalSEDv3':
        
        # Global features are reduced to a 16 dimensional vector with MLP, 
        # Reduced representation is then appended to the decoder and encoder inputs!
        
        encoder = EncoderRNN(input_size=14+16, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers,  batch_size=batch_size, dropout = args.dropout).to(device)
        decoder = DecoderRNN(input_size=15+16, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers,  output_size=1, batch_size=batch_size, dropout=args.dropout).to(device)
        network = Net_global_SEDv3(encoder,decoder, output_length, args.dropout, global_feature_size,  device).to(device)

    elif networkType == 'NetglobalSEDv4':

        # Global features are reduced to a 16 dimensional vector with a *CNN*, 
        # Reduced representation is then appended to the the current GRU hidden state.
        # calibrated output = FC([current GRU hidden state,raw global features])
       
        # Important: Global features has to be a 1 dimensional vector for compatibility with the CNN.
        
        globalReducer = globalReducer(batch_size = batch_size, dropout=args.dropout, global_feature_size=global_feature_size, reducerType='CNN_pm25',device=device)
        encoder = EncoderRNN_withGlobal_v2(input_size=14, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size,GlobalReducer=globalReducer,device=device).to(device)
        decoder = DecoderRNN_withGlobal_v2(input_size=15, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, output_size=1, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size,GlobalReducer=globalReducer,device=device).to(device)
        network = Net_global_SEDv4(encoder,decoder, output_length, args.dropout, device).to(device)
        
    elif networkType == 'NetglobalSEDv5':
        
        # Global features are reduced to a 16 dimensional vector with *MLP*, 
        # Reduced representation is then appended to the decoder and encoder inputs!
        # add FC/CNN to global features to NetglobalSEDv1
        
        globalReducer = globalReducer(batch_size = batch_size, dropout=args.dropout, global_feature_size=global_feature_size, reducerType='MLP',device=device)
        encoder = EncoderRNN_withGlobal_v2(input_size=14, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size,GlobalReducer=globalReducer,device=device).to(device)
        decoder = DecoderRNN_withGlobal_v2(input_size=15, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, output_size=1, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size,GlobalReducer=globalReducer,device=device).to(device)
        # This is not a typo, Net_global_SEDv4 backbone is intended to be used for different globalReducer architectures.
        network = Net_global_SEDv4(encoder,decoder, output_length, args.dropout, device).to(device)
        

    elif networkType == 'NetglobalSEDv6':
        # Global features are reduced to a 16 dimensional vector with a *CNN*, 
        # Reduced representation is then appended to the the current GRU hidden state.
        # calibrated output = FC([current GRU hidden state,raw global features])
       
        # Important: Global features has to be a 1 dimensional vector for compatibility with the CNN.
        # Note: Only difference from NetglobalSEDv4 is the CNN architecture. 
        #       To change the CNN params, define your architecture in globalReducer class with a unique reducerType.
        
        globalReducer = globalReducer(batch_size = batch_size, dropout=args.dropout, global_feature_size=global_feature_size, reducerType='CNNv2_pm25',device=device)
        encoder = EncoderRNN_withGlobal_v3(input_size=14, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size,GlobalReducer=globalReducer,device=device).to(device)
        decoder = DecoderRNN_withGlobal_v3(input_size=15, hidden_size=args.hiddenSize, num_grulstm_layers=args.numLayers, output_size=1, batch_size=batch_size, dropout=args.dropout, global_feature_size=global_feature_size,GlobalReducer=globalReducer,device=device).to(device)
        # This is not a typo, Net_global_SEDv4 backbone is intended to be used for different globalReducer architectures.
        network = Net_global_SEDv4(encoder,decoder, output_length, args.dropout, device).to(device)
                 

    # Possible model ideas:
        
    #elif networkType == 'NetglobalSEDv6': unsupervised in the v1 sense
    #elif networkType == 'NetglobalSEDv7': (maybe) experiment with attention with global features
    #elif networkType == 'NetglobalSEDv8': add FC/CNN to global features to NetglobalSEDv1
    
    
    # The path to save the model and the scalers.
    path = 'saved_models/'+ ds_name +'/'+ networkType+ '_inputLength'+str(args.inputLength)+'_outputLength'+str(args.outputLength)+'_batchSize'+str(args.batchSize)+'_epochs'+str(args.epochs)

   
    if not os.path.exists(path):
        os.mkdir(path)
    
    
    modelSaver = ModelSaver(path,args)
    modelSaver.saveScalers(scalerX, scalerY, scalerX_global)
    
    
    # Start training!
    print("Starting the training!")
    train_model(network,trainloader,validationloader,loss_type=args.loss,learning_rate=args.lr, epochs=args.epochs, gamma=1, print_every=1, eval_every=1, verbose=1, scalerY=scalerY, modelSaver=modelSaver, device=device)
    
    
    # Let the garbage collector free the memory   
    trainloader  = []
    Xg_Enc_train = []
    Xg_Dec_train = []
    Xg_Enc_val   = []
    Xg_Dec_val   = []
    X_global     = []
    
     
    # Load the saved best model and do the monthly evaluation.
    with open(modelSaver.model_path, 'rb') as f:
        net = torch.load(f)
        
    evaluate_model_monthly(s1_df, net,scalerX, scalerY, scalerX_global, input_length, output_length, global_feature_length, global_feature_type)
    print('Best epoch: ', modelSaver.best_epoch )

    
    labels,predicts = predict(network,validationloader,device,scalerY)
    l,p,means,stds  = plotSequence(labels,predicts)
    

    # For prediction in general follow these steps:
    #   1) Initialize the dataset, which must be an instance of utils_new.customDataset class.
    #   2) Initialize the utils_new.DataLoader with the customDataset in 1
    #   3) Load a network with torch.load
    #   4) Call predict

    # trainloader     = DataLoader(dataset_train, batch_size=batch_size,shuffle=False, num_workers=1,drop_last=True)
    # labels,predicts = predict(net_gru,trainloader,device)
    # l,p,means,stds  = plotSequence(labels,predicts)
    # np.save('means_train.npy',means)
    # np.save('stds_train.npy',stds)
    # np.save('labels_train.npy',l)
    




    
    
    
    
    
    
    
    


