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

import torch
import torch.nn as nn
import torch.nn.functional as F



class EncoderRNN(torch.nn.Module):
    '''
    Encoder GRU for the default NetSED, no global features.
    '''     
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, 1)
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, input, hidden): 
    
        # input size = [batch_size, 1, input_size]
        output, hidden = self.gru(input, hidden)
        output = F.relu(output)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        
        return output, hidden
    
    def init_hidden(self,device):
        # To initialize hidden state of the GRU with zeros,
        # returns a zeros array with the size of hidden state =[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):   
    '''
    Decoder GRU for the default NetSED, no global features.
    '''  
    def __init__(self, input_size, hidden_size, num_grulstm_layers, output_size,batch_size,dropout):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)
        self.num_grulstm_layers = num_grulstm_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        
    def forward(self, input, hidden):
        
        # input size = [batch_size, 1, input_size]
        output, hidden = self.gru(input, hidden)
        output = F.relu(output)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        
        return output, hidden
    
    def init_hidden(self,device):
        # To initialize hidden state of the GRU with zeros,
        # returns a zeros array with the size of hidden state =[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
           
class Net_SED(nn.Module):
    '''
    The default seq2seq model NetSED, no global features.
    Encoder and decoder should be instances of model.EncoderRNN and model.DecoderRNN, respectively.
    ''' 
    def __init__(self, encoder, decoder, target_length, dropout, device):
        super(Net_SED, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        
        self.dropout=nn.Dropout(dropout)
        self.BN_1 = nn.BatchNorm1d(decoder.input_size)
        self.BN_layers = nn.ModuleList()
        for i in range(target_length):
            self.BN_layers.append(nn.BatchNorm1d(decoder.input_size))

        
    def forward(self, X_Encoder, X_Decoder, Xg_Encoder, Xg_Decoder):
        
        #Xg_Encoder and Xg_Decoder(i.e. global features) are not used in this network
        #X_Encoder, X_Decoder contains splitted input and output sequences.
        
        input_length  = X_Encoder.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        
        #Loop over input sequence and update the hidden state.
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(X_Encoder[:,ei:ei+1,:], encoder_hidden)
            
        #Concatenate 1 dimensional encoder_output(i.e. last calibrated value) to X_Decoder -> input_channels+1 dimensional   
        #So the first input to decoder GRU cell is:
        decoder_input    = torch.cat(( X_Decoder[:,0:1,:], encoder_output), 2)
        
        #Apply batch norm:
        decoder_input    = decoder_input.permute(0,2,1)
        decoder_input    = self.BN_1(decoder_input)
        decoder_input    = decoder_input.permute(0,2,1)
        
        #Apply dropout:
        decoder_input    = self.dropout(decoder_input)
        
        #Initialize decoder hidden state with the last encoder hidden state.
        decoder_hidden = encoder_hidden
        
        outputs = torch.zeros([X_Decoder.shape[0], self.target_length, 1]  ).to(self.device)
        
        #Loop over output sequence, update the hidden state and calibrate each point.
        for di in range(self.target_length):
            
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            #There is no input after the last iteration
            if(di+1 != self.target_length):
                
                # input = [measurements , last calibrated value]
                decoder_input   = torch.cat(( X_Decoder[:,di+1:di+2,:], decoder_output), 2)
                
                #Batch norm
                decoder_input   = decoder_input.permute(0,2,1)
                BN_temp         = self.BN_layers[di]
                decoder_input   = BN_temp(decoder_input)
                decoder_input   = decoder_input.permute(0,2,1)
                
                #Dropout
                decoder_input = self.dropout(decoder_input)
                
            #Calibrated outputs
            outputs[:,di:di+1,:] = decoder_output
    
        return outputs      


class EncoderRNN_withGlobal(torch.nn.Module):
    '''
    Encoder GRU for NetglobalSEDv1  with the global features.
    Raw global features are appended to the current GRU hidden state.
    Calibrated output = FC([current GRU hidden state,raw global features])
    '''   
    
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size, dropout, global_feature_size):
        super(EncoderRNN_withGlobal, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.global_feature_size = global_feature_size
        self.dropout=nn.Dropout(dropout)
        
        #Main difference:
        self.out = nn.Linear(global_feature_size+hidden_size, 1)
                
    def forward(self, input, hidden, global_input): # input [batch_size, length T, dimensionality d]
        output, hidden = self.gru(input, hidden)
        
        output = F.relu(output)
        output = torch.cat(( output, global_input), 2)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        
        return output, hidden
    
    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
    
class DecoderRNN_withGlobal(nn.Module):
    '''
    Decoder GRU for NetglobalSEDv1  with the global features.
    Raw global features are appended to the current GRU hidden state.
    Calibrated output = FC([current GRU hidden state,raw global features])
    '''    
    
    def __init__(self, input_size, hidden_size, num_grulstm_layers, output_size,batch_size,dropout, global_feature_size):
        super(DecoderRNN_withGlobal, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.num_grulstm_layers = num_grulstm_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.global_feature_size = global_feature_size
        #Main difference:
        self.out = nn.Linear(global_feature_size+hidden_size, output_size)
        
    def forward(self, input, hidden, global_input):
        output, hidden = self.gru(input, hidden)
 
        output = F.relu(output)
        output = torch.cat(( output, global_input), 2)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        
        return output, hidden
    
    def init_hidden(self,device):
            #[num_layers*num_directions,batch,hidden_size]
            return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
           
            
class Net_global_SEDv1(nn.Module):
    '''
    Seq2seq model NetglobalSEDv1 with global features.
    Global features are appended to current GRU hidden state in decoder and encoder to form a joint representation.
    The final calibrated output is obtained by passing the joing representation through a perceptron layer.
    
    Underlying architecture is the same as NetSED but with two differences:
        1)Xg_Encoder, Xg_Decoder inputs of the forward prop is utilized here.
        2)Different encoder and decoder. 
    '''   
    
    #global features are appended to the decoder and encoder hidden state before calibration
    def __init__(self, encoder, decoder, target_length, dropout, device):
        super(Net_global_SEDv1, self).__init__()
        #encoder and decoder must be instances of EncoderRNN_withGlobal and DecoderRNN_withGlobal, respectively 
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        self.dropout=nn.Dropout(dropout)
        self.BN_1 = nn.BatchNorm1d(decoder.input_size)
        self.BN_layers = nn.ModuleList()
        for i in range(target_length):
            self.BN_layers.append(nn.BatchNorm1d(decoder.input_size))

        
    def forward(self, X_Encoder, X_Decoder, Xg_Encoder, Xg_Decoder):
        
        input_length  = X_Encoder.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(X_Encoder[:,ei:ei+1,:], encoder_hidden, Xg_Encoder[:,ei:ei+1] )
            
        #Concatenate 1 dimensional encoder_output(i.e. last calibrated value) to X_Decoder -> input_channels+1 dimensional   
        #So the first input to decoder GRU cell is:
        decoder_input    = torch.cat(( X_Decoder[:,0:1,:], encoder_output), 2)
        
        decoder_input    = decoder_input.permute(0,2,1)
        decoder_input    = self.BN_1(decoder_input)
        decoder_input    = decoder_input.permute(0,2,1)
        
        decoder_input    = self.dropout(decoder_input)
        
        
        decoder_hidden = encoder_hidden
        outputs = torch.zeros([X_Decoder.shape[0], self.target_length, 1]  ).to(self.device)
        
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, Xg_Decoder[:,di:di+1,:])
            #decoder_output_scaled = F.sigmoid(decoder_output)
            if(di+1 != self.target_length):
                decoder_input   = torch.cat(( X_Decoder[:,di+1:di+2,:], decoder_output), 2)
                
                decoder_input   = decoder_input.permute(0,2,1)
                BN_temp         = self.BN_layers[di]
                decoder_input   = BN_temp(decoder_input)
                decoder_input   = decoder_input.permute(0,2,1)
                
                decoder_input = self.dropout(decoder_input)
                #There is no input after the last iteration
            outputs[:,di:di+1,:] = decoder_output
    
    
        return outputs
    

class Net_global_SEDv2(nn.Module):
    
    '''
    Seq2seq model NetglobalSEDv2 with global features.
    Global features are appended to current GRU *input* in decoder and encoder.
    
    Underlying architecture is the same as NetSED but with one difference:
        1)Xg_Encoder, Xg_Decoder inputs of the forward prop is appended to X_Encoder and X_Decoder, respectively.
    '''   
    #global features are appended to the decoder and encoder inputs!
    def __init__(self, encoder, decoder, target_length, dropout, device):
        super(Net_global_SEDv2, self).__init__()
        #encoder and decoder must be instances of EncoderRNN and DecoderRNN, respectively 
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        
        self.dropout=nn.Dropout(dropout)
        self.BN_1 = nn.BatchNorm1d(decoder.input_size)
        self.BN_layers = nn.ModuleList()
        for i in range(target_length):
            self.BN_layers.append(nn.BatchNorm1d(decoder.input_size))


        
    def forward(self, X_Encoder, X_Decoder, Xg_Encoder, Xg_Decoder):
        
        input_length  = X_Encoder.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        
        #the global features are appendded to encoder input
        encoder_input = torch.cat((X_Encoder,Xg_Encoder),2)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(encoder_input[:,ei:ei+1,:], encoder_hidden)
            
        #concat 1D encoder_output with X_Decoder -> 15dimensional
        #first decoder input
        
        decoder_input_with_global = torch.cat((X_Decoder,Xg_Decoder),2)
        decoder_input    = torch.cat(( decoder_input_with_global[:,0:1,:], encoder_output), 2)
        
        decoder_input    = decoder_input.permute(0,2,1)
        decoder_input    = self.BN_1(decoder_input)
        decoder_input    = decoder_input.permute(0,2,1)
        
        decoder_input    = self.dropout(decoder_input)
        
        
        decoder_hidden = encoder_hidden
        outputs = torch.zeros([X_Decoder.shape[0], self.target_length, 1]  ).to(self.device)
        
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            #decoder_output_scaled = F.sigmoid(decoder_output)
            if(di+1 != self.target_length):
                decoder_input   = torch.cat(( decoder_input_with_global[:,di+1:di+2,:], decoder_output), 2)
                
                decoder_input   = decoder_input.permute(0,2,1)
                BN_temp         = self.BN_layers[di]
                decoder_input   = BN_temp(decoder_input)
                decoder_input   = decoder_input.permute(0,2,1)
                
                decoder_input = self.dropout(decoder_input)
                #There is no input after the last iteration
            outputs[:,di:di+1,:] = decoder_output
    
    
        return outputs
    

class Net_global_SEDv3(nn.Module):
    
    '''
    Seq2seq model NetglobalSEDv2 with global features.
    Global features reduced to 16 dimensions with a MLP, then appended to current GRU *input* in decoder and encoder.
    '''   
    
    #global features are fed to a MLP, which are then appended to the decoder and encoder inputs!
    def __init__(self, encoder, decoder, target_length, dropout, global_length, device):
        super(Net_global_SEDv3, self).__init__()
        #encoder and decoder must be instances of EncoderRNN and DecoderRNN, respectively 
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        
        self.dropout=nn.Dropout(dropout)
        self.BN_1 = nn.BatchNorm1d(decoder.input_size)
        self.BN_layers = nn.ModuleList()
        
        self.reducedLength = 16
        self.fc_XgEncoder = nn.Linear(global_length, self.reducedLength)
        self.fc_XgDecoder = nn.Linear(global_length, self.reducedLength)
        

        for i in range(target_length):
            self.BN_layers.append(nn.BatchNorm1d(decoder.input_size))


        
    def forward(self, X_Encoder, X_Decoder, Xg_Encoder, Xg_Decoder):
        
        input_length  = X_Encoder.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        
        Xg_Encoder_reduced = torch.zeros([Xg_Encoder.shape[0],Xg_Encoder.shape[1],self.reducedLength],device=self.device)
        
        for i in range(Xg_Encoder.shape[0]):
            for j in range(Xg_Encoder.shape[1]):
                 Xg_Encoder_reduced[i,j,:] = self.fc_XgEncoder(Xg_Encoder[i,j,:])
                 
        #the global features are appendded to encoder input
        encoder_input = torch.cat((X_Encoder,Xg_Encoder_reduced),2)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(encoder_input[:,ei:ei+1,:], encoder_hidden)
            
        #concat 1D encoder_output with X_Decoder -> 15dimensional
        #first decoder input
        
        
        Xg_Decoder_reduced = torch.zeros([Xg_Decoder.shape[0],Xg_Decoder.shape[1],self.reducedLength],device=self.device)
        
        for i in range(Xg_Decoder.shape[0]):
            for j in range(Xg_Decoder.shape[1]):
                 Xg_Decoder_reduced[i,j,:] = self.fc_XgDecoder(Xg_Decoder[i,j,:])
        
        decoder_input_with_global = torch.cat((X_Decoder,Xg_Decoder_reduced),2)
        decoder_input    = torch.cat(( decoder_input_with_global[:,0:1,:], encoder_output), 2)
        
        decoder_input    = decoder_input.permute(0,2,1)
        decoder_input    = self.BN_1(decoder_input)
        decoder_input    = decoder_input.permute(0,2,1)
        
        decoder_input    = self.dropout(decoder_input)
        
        
        decoder_hidden = encoder_hidden
        outputs = torch.zeros([X_Decoder.shape[0], self.target_length, 1]  ).to(self.device)
        
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            #decoder_output_scaled = F.sigmoid(decoder_output)
            if(di+1 != self.target_length):
                decoder_input   = torch.cat(( decoder_input_with_global[:,di+1:di+2,:], decoder_output), 2)
                
                decoder_input   = decoder_input.permute(0,2,1)
                BN_temp         = self.BN_layers[di]
                decoder_input   = BN_temp(decoder_input)
                decoder_input   = decoder_input.permute(0,2,1)
                
                decoder_input = self.dropout(decoder_input)
                #There is no input after the last iteration
            outputs[:,di:di+1,:] = decoder_output
    
    
        return outputs
    
    

class globalReducer(torch.nn.Module):
    '''
    Generalized reducer module for global features. The subnetwork defined here processes the global features
    and reduces its dimensionality. For defining different types of reducers, modify the __init__()
    '''       
    
    def __init__(self,batch_size, dropout, global_feature_size, reducerType,device):
    
        super(globalReducer,self).__init__()
        self.reducerType = reducerType
        self.global_feature_size = global_feature_size
        self.device = device

        if reducerType=='MLP':
            #Can be used with any type of global features as long as you flatten them first.
            self.reducer = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(in_features=global_feature_size, out_features=128, bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         
                                         nn.Linear(in_features=128, out_features=128, bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         
                                         nn.Linear(in_features=128, out_features=128, bias=False),
                                         nn.ReLU(inplace=True)
                                         )
        elif reducerType=='CNN_pm25':
            #Intended to be used with raw pm2.5 values as global features.
            self.reducer = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=128,kernel_size=[8]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=128, out_channels=64, kernel_size=[8]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=64, out_channels=16, kernel_size=[8]),
                                         nn.ReLU(inplace=True),                                         
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=16, out_channels=1, kernel_size=[8]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2])                                         
                            
                                         )  
        elif reducerType=='CNNv2_pm25':
            #Intended to be used with raw pm2.5 values as global features.
            self.reducer = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32,kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=32, out_channels=64, kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=64, out_channels=128, kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=128, out_channels=1, kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2])
                            
                                         )

        elif reducerType=='CNN_all':
            #Intended to be used with raw global features containing all input channels.
            self.reducer = nn.Sequential(nn.Conv1d(in_channels=14, out_channels=128,kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=128, out_channels=64, kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=64, out_channels=32, kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2]),
                                         
                                         nn.Conv1d(in_channels=32, out_channels=1, kernel_size=[4]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout),
                                         nn.MaxPool1d(kernel_size=[2])
                            
                                         )
            
        
    def forward(self, Xg):
        return self.reducer(Xg)

    def get_output_shape(self, b_size,global_length):
        #Returns the output shape of the selected globalReducer architecture.
        
        if self.reducerType=='MLP':
            image_dim = [b_size,4,global_length]
        elif self.reducerType=='CNN_pm25':
            image_dim = [b_size,1,global_length]
        elif self.reducerType=='CNNv2_pm25':
            image_dim = [b_size,1,global_length]
          
        o_size = self.forward(torch.rand(*(image_dim)).to(self.device)).data.shape     
        return o_size[-1]
      
# =============================================================================
# Encoder and Decoder for the network with the global features appended at the last FC layer
# MLP and CNN extension for the global features
# =============================================================================

class EncoderRNN_withGlobal_v2(torch.nn.Module):
    '''
    Encoder for the network with the global feature appended to the current hidden state.
    Global features are first processed with an instance of globalReducer.
    '''   
    
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size, dropout, global_feature_size, GlobalReducer,device):
        super(EncoderRNN_withGlobal_v2, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.global_feature_size = global_feature_size
        self.dropout=nn.Dropout(dropout)

        #Main difference:
            
        self.GlobalReducer = GlobalReducer.to(device)
        self.reducedShape = GlobalReducer.get_output_shape(batch_size,global_feature_size)
        self.out = nn.Linear(self.reducedShape+hidden_size, 1)
        
                
    def forward(self, input, hidden, global_input): # input [batch_size, length T, dimensionality d]
        output, hidden = self.gru(input, hidden)
        
        global_reduced = self.GlobalReducer(global_input)
        
        output = F.relu(output)
        output = torch.cat(( output, global_reduced), 2)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        
        return output, hidden
    
    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
    
class DecoderRNN_withGlobal_v2(nn.Module):
    '''
    Decoder for the network with the global feature appended to the current hidden state.
    Global features are first processed with an instance of globalReducer.
    '''    
    def __init__(self, input_size, hidden_size, num_grulstm_layers, output_size,batch_size,dropout, global_feature_size, GlobalReducer,device):
        super(DecoderRNN_withGlobal_v2, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.num_grulstm_layers = num_grulstm_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.global_feature_size = global_feature_size
        
        #Main difference:
        self.GlobalReducer = GlobalReducer.to(device)
        self.reducedShape = GlobalReducer.get_output_shape(batch_size,global_feature_size)
        self.out = nn.Linear(self.reducedShape+hidden_size, 1)
        
    def forward(self, input, hidden, global_input):
        output, hidden = self.gru(input, hidden)
 
        global_reduced = self.GlobalReducer(global_input)
            
        output = F.relu(output)
        output = torch.cat(( output, global_reduced), 2)
        output = self.dropout(output)
        output = F.relu(self.out(output))
        
        return output, hidden
    
    def init_hidden(self,device):
            #[num_layers*num_directions,batch,hidden_size]
            return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)   

class Net_global_SEDv4(nn.Module):
    '''
    Generalized NetglobalSED with the global feature appended to the current hidden state.
    Global features are first processed with an instance of globalReducer.
    '''  

    def __init__(self, encoder, decoder, target_length, dropout, device):
        super(Net_global_SEDv4, self).__init__()
        #encoder and decoder must be instances of EncoderRNN_withGlobal_v2 and DecoderRNN_withGlobal_v2, respectively 
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device
        
        self.dropout=nn.Dropout(dropout)
        self.BN_1 = nn.BatchNorm1d(decoder.input_size)
        self.BN_layers = nn.ModuleList()
        for i in range(target_length):
            self.BN_layers.append(nn.BatchNorm1d(decoder.input_size))


        
    def forward(self, X_Encoder, X_Decoder, Xg_Encoder, Xg_Decoder):
        
        #global features are appended to the decoder and encoder hidden state before calibration
        #Extracted using globalReducer class
        
        input_length  = X_Encoder.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(X_Encoder[:,ei:ei+1,:], encoder_hidden, Xg_Encoder[:,ei:ei+1] )
            
        #concat 1D encoder_output with X_Decoder -> 15dimensional
        #first decoder input
        decoder_input    = torch.cat(( X_Decoder[:,0:1,:], encoder_output), 2)
        
        decoder_input    = decoder_input.permute(0,2,1)
        decoder_input    = self.BN_1(decoder_input)
        decoder_input    = decoder_input.permute(0,2,1)
        
        decoder_input    = self.dropout(decoder_input)
        
        
        decoder_hidden = encoder_hidden
        outputs = torch.zeros([X_Decoder.shape[0], self.target_length, 1]  ).to(self.device)
        
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, Xg_Decoder[:,di:di+1,:])
            #decoder_output_scaled = F.sigmoid(decoder_output)
            if(di+1 != self.target_length):
                decoder_input   = torch.cat(( X_Decoder[:,di+1:di+2,:], decoder_output), 2)
                
                decoder_input   = decoder_input.permute(0,2,1)
                BN_temp         = self.BN_layers[di]
                decoder_input   = BN_temp(decoder_input)
                decoder_input   = decoder_input.permute(0,2,1)
                
                decoder_input = self.dropout(decoder_input)
                #There is no input after the last iteration
            outputs[:,di:di+1,:] = decoder_output
    
    
        return outputs
    

class EncoderRNN_withGlobal_v3(torch.nn.Module):
    '''
    Encoder for the networks with the global feature appended to the current hidden state.
    Global features are first processed with an instance of globalReducer. Then passed through a Linear layer to reduce the dimensionality to 16.
    '''   
    
    def __init__(self,input_size, hidden_size, num_grulstm_layers, batch_size, dropout, global_feature_size, GlobalReducer,device):
        super(EncoderRNN_withGlobal_v3, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_grulstm_layers = num_grulstm_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.global_feature_size = global_feature_size
        self.dropout=nn.Dropout(dropout)

        #Main difference:
            
        self.GlobalReducer = GlobalReducer.to(device)
        self.reducedShape = GlobalReducer.get_output_shape(batch_size,global_feature_size)
        self.out1 = nn.Linear(self.reducedShape+hidden_size, 16)
        self.out2 = nn.Linear(16,1)
                
    def forward(self, input, hidden, global_input): # input [batch_size, length T, dimensionality d]
        output, hidden = self.gru(input, hidden)
        
        global_reduced = self.GlobalReducer(global_input)
        
        output = F.relu(output)
        output = torch.cat(( output, global_reduced), 2)
        output = self.dropout(output)
        output = F.relu(self.out1(output))
        output = F.relu(self.out2(output))       
        
        return output, hidden
    
    def init_hidden(self,device):
        #[num_layers*num_directions,batch,hidden_size]
        return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)
    
class DecoderRNN_withGlobal_v3(nn.Module):
    '''
    Decoder for the networks with the global feature appended to the current hidden state.
    Global features are first processed with an instance of globalReducer. Then passed through a Linear layer to reduce the dimensionality to 16.
    '''   
    def __init__(self, input_size, hidden_size, num_grulstm_layers, output_size,batch_size,dropout, global_feature_size, GlobalReducer,device):
        super(DecoderRNN_withGlobal_v3, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_grulstm_layers,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.num_grulstm_layers = num_grulstm_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.global_feature_size = global_feature_size
        
        #Main difference:
        self.GlobalReducer = GlobalReducer.to(device)
        self.reducedShape = GlobalReducer.get_output_shape(batch_size,global_feature_size)
        self.out1 = nn.Linear(self.reducedShape+hidden_size, 16)
        self.out2 = nn.Linear(16,1)
        
    def forward(self, input, hidden, global_input):
        output, hidden = self.gru(input, hidden)
 
        global_reduced = self.GlobalReducer(global_input)
            
        output = F.relu(output)
        output = torch.cat(( output, global_reduced), 2)
        output = self.dropout(output)
        output = F.relu(self.out1(output))
        output = F.relu(self.out2(output))    
        
        return output, hidden
    
    def init_hidden(self,device):
            #[num_layers*num_directions,batch,hidden_size]
            return torch.zeros(self.num_grulstm_layers, self.batch_size, self.hidden_size, device=device)   
