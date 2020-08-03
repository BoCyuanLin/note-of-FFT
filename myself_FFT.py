# -*- coding: utf-8 -*-
"""
Class: myself_FFT
date: 2019/12/20
version: 1.0.0
"""
import numpy as np
import math

class myself_FFT:        
    
    
    def instructions(self):
        print('Class: Myself implement of FFT \n')
        print('Version: 1.0.0 \n')
        print('Functions\n', 
              '  omega(N,K): calculate the complex root with order N and variable k\n',
              '  FFT_1D(data): calculate the $1-$ dimension FFT of input data\n',
              '  ')
        
    # complex root calculator
    def omega(self, N, k):
        return np.exp(((2.0)*np.pi*1j*k)/N)
    
    # compute 1-D FFT
    def FFT_1D(self, data):
    
        N = len(data) # Catch the length of the data
    
        # If the length of data is 1, return the data derictly
        if N == 1:
            return data
    
        # Divide the sequences into even and odd part, then (Divide & Conquer)
        deven, dodd = self.FFT_1D(data[0::2]), self.FFT_1D(data[1::2]) 
    
        # Generate the complex array to save the FFT
        result = np.zeros(data.shape, complex) 
    
        omg = 0
        # implement the butterfly algorithm
        for i in range(int(N/2)):
            
            omg = self.omega(N,-i)*dodd[i]
        
            result[i] = deven[i] + omg
        
            result[i+int(N/2)] = deven[i] - omg
    
        return result
    
    # compute 1-D inverse FFT
    def FFT_1D_INV(self, Fdata):
        # By the Section3 in my note (Inverse Discrete Fourier Transform Processing)
        return np.conj(self.FFT_1D(np.conj(Fdata)))/len(Fdata)
    
    # pad the input 2-D data into shape (2-power,)
    def Padding_1D(self, data):
        
        length = len(data)
        
        L = 2**int(math.ceil(math.log(length, 2)))
        
        Pdata = np.zeros((L,), dtype = data.dtype)
        
        Pdata[:length] = data
        
        return Pdata
    
    def Delete_Padding_1D(self, data, origlen):
    
        # Generate the blank array with original length
        blank = np.zeros(origlen, np.uint8) 
    
        blank = data[0:origlen] # Delete the padding data
    
        return blank
    
    
    # Since the linear property of the 2D DFT, we can calculate 1D FFT with each rows,
    #------ then use the result to compute 1D FFT again with each column.
    # compute 2-D FFT
    def FFT_2D(self, data):
    
        Pdata, H, W = self.Padding_2D(data)
    
        return np.transpose(self.FFT_1D(np.transpose(self.FFT_1D(Pdata)))), H, W
    
    # compute the 2-D inverse FFT
    def FFT_2D_INV(self,Fdata, h, w):
    
        idata, H, W = self.FFT_2D(np.conj(Fdata))
    
        idata = np.array(np.real(np.conj(idata)))/(H*W)
    
        return idata[0:h, 0:w]
    
    # shift the low frequency into center
    def FFT_Shift(self, Fdata):
        H, W = Fdata.shape
        h, w = int(H/2), int(W/2)
        B1, B2 = Fdata[0: h, 0: w], Fdata[h: H, 0: w]
        B3, B4 = Fdata[0: h, w: W], Fdata[h: H, w: W]
        Fswitch = np.zeros(Fdata.shape, Fdata.dtype)
        Fswitch[h: H, w: W], Fswitch[0: h, 0: w] = B1, B4
        Fswitch[h: H, 0: w], Fswitch[0: h, w: W]= B3, B2
        return Fswitch
    
    # pad the input 2-D data into shape (2-power, 2-power)
    def Padding_2D(self, data):
    
        h, w = np.shape(data)
    
        H, W = 2**int(math.ceil(math.log(h, 2))), 2 ** int(math.ceil(math.log(w, 2)))
    
        Pdata = np.zeros((H,W), dtype = data.dtype)
    
        Pdata[0:h, 0:w] = data
    
        return Pdata, H, W
    

