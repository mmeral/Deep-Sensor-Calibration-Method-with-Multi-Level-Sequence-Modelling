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

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pickle import dump



def split_dataset(df, input_length, output_length, global_feature_length,global_feature_type):
    
    '''
    Split the dataset into input and output sequences of sequence to sequence models.

    :param pandas.DataFrame df: The dataframe containing the measurements
                               -The last column must be the ground truth
                               -The 5th column must be the low-cost-sensor data, not including the recording time

    :param int input_length           : Length of the short term input sequence of the seq2seq like models 
    :param int output_length          : Length of the output sequence of the seq2seq like models 
    :param int global_feature_length  : Number of weeks to be considered as the global history
    :param str global_feature_type    : The type of the extracted global features. 
                                        Allowed values:
                                            pm25_only       : Global features consist of only raw pm2.5 values of the low cost sensor.
                                            ts_features     : Global features consist of weekly max,min,median and std of all measurement channels.
                                            all_channels_raw: Global features consist of raw values of all measurement channels.
   

    :return: np.ndarray X_Enc: Array containing input sequence samples, X_Enc.size = [number of samples,input_length,number of measurement channels in df]
    :return: np.ndarray X_Dec: Array containing output sequence samples, X_Dec.size = [number of samples,output_length,number of measurement channels in df]
    :return: np.ndarray y    : Array containing output ground truth sequence samples, y.size = [number of samples,output_length,1]
    :return: np.ndarray X_global : Array containing global history of the samples, size depends on global_feature_type
    :return: np.ndarray time_idx : Array of strings, containing the recording time of the last value of the output y.
    '''
   
    #Convert weeks to hours, 1 week is 168 hours
    input_length_global = 168*global_feature_length 
    
    
    df_X    = df.drop('time',1)
    df_X    = df_X.drop('pm25_station',1)
    df_gt   = df[df.columns[-1]] #reference station value
    df_time = df[df.columns[0]]  #recording times

    X_original = df_X.to_numpy()
    y_original = df_gt.to_numpy()
    time_original = df_time.to_numpy()
    
    #max_input_length : the index of first calibratable model. 
    #                   The first point in the input sequence must be input_length_global'th sample.
    
    max_input_length = input_length+input_length_global
    
    y_index_END = max_input_length+output_length
    y_index_START = y_index_END-output_length
    
    x_index_Decoder_END   = max_input_length+output_length
    x_index_Decoder_START = x_index_Decoder_END-output_length
    
    x_index_Encoder_END   = max_input_length
    x_index_Encoder_START = max_input_length-input_length
    
    x_global_END   = input_length_global
    x_global_start = 0
    
    X_splitted_Encoder = []
    X_splitted_Decoder = []
    y_splitted         = []
    X_global           = []
    time_idx_splitted  = []
    
    #Extract the global features
    while(x_global_END<y_original.shape[0]):
        
        X_global_temp = []
        
        if(global_feature_type=='pm25_only'):
            for i in range(global_feature_length):
                #raw values
                X_global_temp.extend(X_original[x_global_start+i*168:x_global_start+(i+1)*168, 4])

        elif(global_feature_type=='ts_features'):
            for i in range(global_feature_length):
                X_global_temp.extend(np.mean(X_original[x_global_start+i*168:x_global_start+(i+1)*168],axis=0))
                X_global_temp.extend(np.max(X_original[x_global_start+i*168:x_global_start+(i+1)*168],axis=0))
                X_global_temp.extend(np.min(X_original[x_global_start+i*168:x_global_start+(i+1)*168],axis=0))
                X_global_temp.extend(np.median(X_original[x_global_start+i*168:x_global_start+(i+1)*168],axis=0))
                
        elif(global_feature_type=='all_channels_raw'):
                
            X_global_temp  =   X_original[x_global_start:x_global_start+input_length_global]
            
        X_global.append(X_global_temp)
        
        x_global_END     += 1
        x_global_start   += 1        

    
    X_global = np.array(X_global)
    
    if(global_feature_type=='all_channels_raw'):
        X_global = np.swapaxes(X_global,1,2)
        
    
    X_splitted_global_Encoder = []
    X_splitted_global_Decoder = []
    global_START = 0

    #Extract the input and output sequences
    while(y_index_END<y_original.shape[0]):
        

        y_splitted.append(y_original[y_index_START:y_index_END])
        time_idx_splitted.append(time_original[y_index_END])
        
        X_splitted_Encoder.append(X_original[x_index_Encoder_START:x_index_Encoder_END])
        X_splitted_Decoder.append(X_original[x_index_Decoder_START:x_index_Decoder_END])
        
        X_splitted_global_Encoder.append(X_global[global_START:global_START+input_length])
        X_splitted_global_Decoder.append(X_global[global_START+input_length:global_START+input_length+output_length])
        

        y_index_START +=1
        y_index_END   +=1

        x_index_Decoder_END   += 1
        x_index_Decoder_START += 1
        
        x_index_Encoder_END   += 1
        x_index_Encoder_START += 1
        
        global_START +=1

    
    X_Enc = np.array(X_splitted_Encoder)
    X_Dec = np.array(X_splitted_Decoder)
    y = np.array(y_splitted)
    X_global = X_global
    Xg_Enc = np.array( X_splitted_global_Encoder )
    Xg_Dec = np.array( X_splitted_global_Decoder )
    time_idx = np.array(time_idx_splitted)
    
    return X_Enc, X_Dec, y, X_global, Xg_Enc, Xg_Dec,time_idx




def monthly_train_val_split(X, X_global, y, time_idx, last_train_month, last_val_month, inputlenth, outputlength):

    '''
    Does the train-validation splitting. The data is assumed to be splitted into sequences with split_dataset method previosly.

    :param np.ndarray X        : Short input sequences
    :param np.ndarray X_global : Global history sequences
    :param np.ndarray y        : ground truth output sequences
    :param np.ndarray time_idx : the recording time of the last point of the sequence y
    :param str last_train_month: The last month in the training set. The last month is completely included.
                                    -String has to be formatted as XXXX-YY, with XXXX=year and YY=month
    :param str last_val_month  : The last month in the validation set. The last month is completely included.
                                    -String has to be formatted as XXXX-YY, with XXXX=year and YY=month                             
    :param int input_length    : Length of the short term input sequence of the seq2seq like models 
    :param int output_length   : Length of the output sequence of the seq2seq like models 


    :return: np.ndarray X_train           : Training set
    :return: np.ndarray X_val             : Validation set
    :return: np.ndarray X_global_train    : Training global features
    :return: np.ndarray X_global_val      : Validation global features
    :return: np.ndarray y_train           : Training ground truth sequences
    :return: np.ndarray y_val             : Validation ground truth sequences
    :return: list train_indicies          : The index of the training samples in X. (The 1st dimension of X is the sample dimension)
    :return: list val_indicies            : The index of the validation samples in X. (The 1st dimension of X is the sample dimension)  
    '''

    X_train_val = X
    X_global_train_val = X_global
    y_train_val = y
    
    
    
    #Calculate the next month of last_train_month, last_val_month for stopping conditions
    
    val_first_year  = int(last_train_month[0:4])
    val_first_month = int(last_train_month[5:7])+1 
    
    #If month>12, happy new year!
    if(val_first_month>12):
        val_first_year = val_first_year +1;
        val_first_month = val_first_month-12;
        
    # Format month to 2 digits, e.g. int(7)->"07" 
    first_val_month = str(val_first_year)+'-'+str(val_first_month).zfill(2)
    
    test_first_year  = int(last_val_month[0:4])
    test_first_month = int(last_val_month[5:7])+1
    
    if(test_first_month>12):
        test_first_year  = test_first_year +1;
        test_first_month = test_first_month-12;
        
    
    first_test_month = str(test_first_year)+'-'+str(test_first_month).zfill(2)


    train_indicies = []
    val_indicies = []
    
    #Loop until the stopping months. first_val_month :stopping month for training
    i = 0;      
    while (time_idx[i][0:7] != first_val_month):
        train_indicies.append(i)
        i+=1
        
    #Loop until the stopping months. first_test_month :stopping month for validation  
    i = train_indicies[-1]+1
    while (time_idx[i][0:7] != first_test_month):
        val_indicies.append(i)
        i+=1   
    
    X_train         = X_train_val[train_indicies,:]
    y_train         = y_train_val[train_indicies,:]
    X_global_train  = X_global_train_val[train_indicies,:]
        
    X_val   = X_train_val[val_indicies,:]    
    y_val   = y_train_val[val_indicies,:]
    X_global_val = X_global_train_val[val_indicies,:]
      
    return X_train, X_val, X_global_train, X_global_val, y_train,y_val,train_indicies,val_indicies


class customDataset(torch.utils.data.Dataset):
    
    '''
    A custom torch Dataset class, a returned sample from the dataset contains 
                                        -encoder input sequence
                                        -decoder input sequence
                                        -encoder global features
                                        -decoder global features
                                        -ground  truth sequence
    '''
    def __init__(self, X_Enc, X_Dec, Xg_Enc, Xg_Dec, y):
        super(customDataset, self).__init__()  
        self.X_Enc = X_Enc
        self.X_Dec = X_Dec
        self.Xg_Enc = Xg_Enc
        self.Xg_Dec = Xg_Dec
        self.y = y
        
    def __len__(self):
        return (self.X_Enc).shape[0]

    def __getitem__(self, idx):
        return (self.X_Enc[idx,:,:], self.X_Dec[idx,:,:], self.Xg_Enc[idx,:,:], self.Xg_Dec[idx,:,:], self.y[idx,:,np.newaxis])
    
    
        
def getScalers(s_df, tr_idx):
    
    '''
        Train a 0-max normalizer for decoder and encoder inputs on the training samples.
        
        :param pandas.DataFrame df: The dataframe containing the measurements
                                   -The last column must be the ground truth
        :param list tr_idx: Training samples indicies
        
        :return: sklearn.preprocessing.MinMaxScaler scalerX: the input normalizer
        :return: sklearn.preprocessing.MinMaxScaler scalerY: the output normalizer
    '''
    
    df_X  = s_df.drop('time',1)
    df_X  = df_X.drop('pm25_station',1)
    df_gt = s_df[s_df.columns[-1]]

    X_original = df_X.to_numpy()
    y_original = df_gt.to_numpy()
    
    X_train = X_original[tr_idx,:]
    y_train = y_original[tr_idx]
    
    #Add an artificial zeros row to training data for training 0-max normalizers
    #The measurements are nonnegative by nature. If a value lower than the minimum in the training set is received,
    #it would be mapped to a negative value. The zero row prevents this happening.
    
    zerosRow = np.zeros((1,X_train.shape[1]))
    zeroRowY = np.zeros((1))
    X_train2 = np.concatenate((zerosRow,X_train),axis=0)
    y_train2 = np.concatenate((zeroRowY,y_train),axis=0)
    
    #Default range 0-1
    #Input normalziation
    scalerX = MinMaxScaler()
    scalerX.fit(X_train2)
    #Output normalziation
    scalerY = MinMaxScaler()
    scalerY.fit(y_train2.reshape(-1,1))
    
    return scalerX, scalerY

def getGlobalScaler(X_global, tr_idx):
    
    '''
        Train a 0-max normalizer for global features on the training samples.
        
        :param np.ndarray X_global: Array containing global features
        :param list tr_idx: Training samples indicies
        
        :return: sklearn.preprocessing.MinMaxScaler scalerX_global: the global feature normalizer
    '''
    
    X_global_train = X_global[tr_idx,:]
    zerosRow = np.zeros((1,X_global_train.shape[1]))
    X_global_train2 = np.concatenate((zerosRow,X_global_train),axis=0)
    
    scalerX_global = MinMaxScaler()
    scalerX_global.fit(X_global_train2)
    
    return scalerX_global
    




def plotSequence(labels,predicts,title="Predicted sequence",plot_raw=False,raw=[]):
    
    '''
        Plots the ground truth, calibrated and raw sequences with a scroll bar. 
        Note:   Since there are multiple calibrated values for each time point due to sequence output and consecutive time points, 
                the mean of the calibrated values are plotted with error bars showing their standard deviation.
        
        
        
        :param list labels  : List of ground truth sequences, each element is a np.ndarray of size (output_length,1)
        :param list predicts: List of calibrated sequences, each element is a np.ndarray of size (output_length,1)
        :param bool plot_raw: Adds raw values specified in raw input to the final plot
        :param list raw     : List of raw sequences, each element is a np.ndarray of size (output_length,1)
        
        :return: list labelSequence    : The ground truth reference measurements, a time series with single channel.
                                         Has the same length as labels, and contains only the last element of each sequence in labels
        :return: list labelPredicts    : The list of calibrated values. For each element in labels, labelPredicts contains a set of calibrated
                                         values due to sequence calibration.
                                            i.e. for a ground truth labelSequence[i], labelPredicts[i] is a list of calibrated values that correspond to
                                            the time point of listlabelSequence[i].
        
        :return: list means : A list containing the mean of every element in labelPredicts
        :return: list stds : A list containing the standard deviation of every element in labelPredicts
        
    '''
    # Labels and predicts are sequences, consequent samples overlap, do not shuffle
    # The order of the sequence has to be preserved in labels and predicts. 
    # To get labels and predicts, you can call predict() defined in main. 
    # Make sure that the shuffle is set to alse for the loader input of predict() 
    
    
    # Flatten and remove overlapping sequences
    labels = labels[:,:]
    predicts = predicts[:,:]
    labelSequence = []
    labelPredicts = []
    
    
    for i in range(labels.shape[0]):
        
        labelSequence.append(labels[i,-1,0])
        predictList = []
        
        for j in range(labels.shape[1]):
            if(i+j>=labels.shape[0]):
                break
            predictList.append(predicts[i+j,labels.shape[1]-1-j,0])
            
        labelPredicts.append(predictList)
            
      
    means = []
    stds = []
    x = np.arange(0,labels.shape[0],1)
    
    plot_window = 100
    
    for lp in labelPredicts:
        means.append(np.mean(lp))
        stds.append(np.std(lp))
    
    
    #Scrollable plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plt.errorbar(x,means,yerr =stds, linestyle='None', marker='^')
    plt.plot(x,labelSequence,label='Reference')
    
    if(plot_raw):
        raw_flat = []
        for i in range(labels.shape[0]):
            raw_flat.append(raw[i,-1])
        plt.plot(x,raw_flat,label='raw')
        
    plt.legend()
    plt.title(title)
    
    axpos = plt.axes([0.2, 0.1, 0.65, 0.03])
    spos = Slider(axpos, 'Pos', 0, labels.shape[0]-plot_window)
   
    y_toplimit = max(means+stds)
    top_gt = max(labelSequence)
    y_toplimit = max(y_toplimit,top_gt)
    
    def update(val):
        pos = spos.val
        ax.axis([pos,pos+plot_window,0,y_toplimit])
        fig.canvas.draw_idle()
        
    spos.on_changed(update)
    plt.show()
    
    return labelSequence,labelPredicts,means,stds


#--------------------------

class ModelSaver():
    
    '''
        A simple class to save the network and preprocessing models on a given path.     
    '''
    
    def __init__(self, path, args):
        
        self.path = path
        self.args = args
        self.best_epoch = 0
        self.best_net = []
        self.model_path = []
        
    def saveNetwork(self,network):
        
        
        model_path = self.path+'/net_.pt'
        with open(model_path, 'wb') as f:
            torch.save(network, f)
            
        self.model_path = model_path
        
    def saveScalers(self, scalerX, scalerY, scalerX_global):
        scalerX_filename = self.path+'/scalerX_.pkl'
        scalerY_filename = self.path+'/scalerY_.pkl'
        scalerGlobal_filename = self.path+'/globalScaler_.pkl'
    
        with open(scalerX_filename,"wb") as f:
            dump(scalerX, f)
            
        with open(scalerY_filename,"wb") as f:
            dump(scalerY, f)
    
        with open(scalerGlobal_filename,"wb") as f:
            dump(scalerX_global, f)

if __name__ == '__main__':
   
    print('This script contains utility functions and not intended to be called as main!')