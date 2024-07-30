import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding
import commpy.channelcoding.turbo as turbo
import commpy.channelcoding.interleavers as RandInterlv
from commpy.utilities import *
import torch
import math
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
from get_args import get_args
from interleavers import Interleaver, DeInterleaver, Trainable_Interleaver

import time
import pickle
args = get_args()
device = torch.device("cuda")

def generate_examples(batch_size = 500, block_len=40 , code_rate = 3):

    trellis1 = cc.Trellis(np.array([2]), np.array([[7,5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7,5]]))

    codewords = np.zeros([1,batch_size,block_len,3])
    true_messages = np.zeros([1,batch_size,block_len,1])

    interleaver = RandInterlv.RandInterlv(block_len,0)

    for example in range(batch_size):

        message_bits = np.random.randint(0, 2, block_len)

        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)
        print(sys.shape)
        print(par1.shape)
        print(par2.shape)

        sys = 2*sys-1
        par1 = 2*par1-1
        par2 = 2*par2-1 

        
  
        codewords[0,example,:,:] = np.concatenate([sys.reshape(block_len,1),par1.reshape(block_len,1),par2.reshape(block_len,1)],axis=1)
        true_messages[0,example,:,:] = message_bits.reshape(block_len,1)


    codewords = codewords.reshape(batch_size,block_len,code_rate)
    true_messages = true_messages.reshape(batch_size,block_len,1)


    return (codewords, true_messages)


def turbolte_encode(inputs):
        
        # Turbo LTE RSC non-systematic part, y[n] = y[n-2] _+ y[n-3] _+ x[n] _+ x[n-1] _+ x[n-3]
        #input_shape = inputs.shape
        X_lte_inputs = torch.zeros((args.batch_size, args.block_len, 1))
        X_lte_inputs = X_lte_inputs.to(device)

        for idx in range(args.block_len):
            if idx == 0:
                xs = inputs[:, idx, :]
                X_lte_inputs[:, idx, :] = xs
            elif idx == 1:
                xs = torch.logical_xor(inputs[:, idx, :], inputs[:, idx - 1, :])
                X_lte_inputs[:, idx, :] = xs

            elif idx ==2:
                xs = torch.logical_xor(inputs[:, idx, :], inputs[:, idx - 1, :])
                X_lte_inputs[:, idx, :] = torch.logical_xor(xs, X_lte_inputs[:, idx - 2, :])

            if idx>=3:
                xs = torch.logical_xor(
                    torch.logical_xor(inputs[:, idx, :], inputs[:, idx - 1, :]), inputs[:, idx - 3, :]
                )

                X_lte_inputs[:, idx, :] = torch.logical_xor(
                    torch.logical_xor(xs, X_lte_inputs[:, idx - 2, :]), X_lte_inputs[:, idx - 3, :]
                )

        return X_lte_inputs


def encoder_yihan(inputs,interleaver):

    inputs_1 = turbolte_encode(inputs)
    inputs_2 = turbolte_encode(interleaver(inputs))
    lte_codes = 2.0 * torch.cat([inputs, inputs_1, inputs_2], dim=2) - 1.0

    return lte_codes





