'''
This module contains all possible encoders, STE, and utilities.
'''

import torch
import torch.nn.functional as F

from numpy import arange
from numpy.random import mtrand
import math
import numpy as np

from interleavers import Interleaver
from utils import snr_db2sigma

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args):

        ctx.save_for_backward(inputs)
        ctx.args = args

        x_lim_abs  = args.enc_value_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if args.enc_quantize_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((args.enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(args.enc_quantize_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.args.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>ctx.args.enc_value_limit]=0
            grad_output[input<-ctx.args.enc_value_limit]=0

        if ctx.args.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.args.enc_grad_limit, ctx.args.enc_grad_limit)

        if ctx.args.train_channel_mode not in ['group_norm_noisy', 'group_norm_noisy_quantize']:
            grad_input = grad_output.clone()
        else:
            # Experimental pass gradient noise to encoder.
            grad_noise = snr_db2sigma(ctx.args.fb_noise_snr) * torch.randn(grad_output[0].shape, dtype=torch.float)
            ave_temp   = grad_output.mean(dim=0) + grad_noise
            ave_grad   = torch.stack([ave_temp for _ in range(ctx.args.batch_size)], dim=2).permute(2,0,1)
            grad_input = ave_grad + grad_noise

        return grad_input, None
        
##############################################
# Encoder Base.
# Power Normalization is implemented here.
##############################################
class ENCBase(torch.nn.Module):
    def __init__(self, args):
        super(ENCBase, self).__init__()

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.reset_precomp()

    def set_parallel(self):
        pass

    def set_precomp(self, mean_scalar, std_scalar):
        self.mean_scalar = mean_scalar.to(self.this_device)
        self.std_scalar  = std_scalar.to(self.this_device)

    # not tested yet
    def reset_precomp(self):
        self.mean_scalar = torch.zeros(1).type(torch.FloatTensor).to(self.this_device)
        self.std_scalar  = torch.ones(1).type(torch.FloatTensor).to(self.this_device)
        self.num_test_block= 0.0

    def enc_act(self, inputs):
        if self.args.enc_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.enc_act == 'elu':
            return F.elu(inputs)
        elif self.args.enc_act == 'relu':
            return F.relu(inputs)
        elif self.args.enc_act == 'selu':
            return F.selu(inputs)
        elif self.args.enc_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.enc_act == 'linear':
            return inputs
        else:
            return inputs

    def power_constraint(self, x_input):

        if self.args.no_code_norm:
            return x_input
        else:
            this_mean    = torch.mean(x_input)
            this_std     = torch.std(x_input)

            
            x_input_norm = (x_input-this_mean)*1.0 / this_std

            if self.args.train_channel_mode == 'block_norm_ste':
                stequantize = STEQuantize.apply
                x_input_norm = stequantize(x_input_norm, self.args)

            return x_input_norm



#######################################################
# TurboAE Encocder, with rate 1/3, CNN-1D same shape only
#######################################################
from cnn_utils import SameShapeConv1d
from cnn_utils import DenseSameShapeConv1d

class ENC_interCNN(ENCBase):
    def __init__(self, args, p_array):
        # turbofy only for code rate 1/3
        super(ENC_interCNN, self).__init__(args)
        self.args             = args

        # Encoder
        if self.args.encoder == 'TurboAE_rate3_cnn':
            self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size
                                                      ,padding_mode=args.padding)

            self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size
                                                      ,padding_mode=args.padding)

            self.enc_cnn_3       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size
                                                      ,padding_mode=args.padding)
        else: # Dense
            self.enc_cnn_1       = DenseSameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

            self.enc_cnn_2       = DenseSameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

            self.enc_cnn_3       = DenseSameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)

        self.interleaver      = Interleaver(args, p_array)



    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs, interleaver):

        inputs     = 2.0*inputs - 1.0
        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int  = interleaver(inputs)
        x_p2       = self.enc_cnn_3(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes

class modulator(ENCBase):
    def __init__(self, args):
        super(modulator, self).__init__(args)
        self.args             = args

        self.mod_linear_1    = torch.nn.Linear(2, 10)
        self.mod_linear_2    = torch.nn.Linear(10, 10)
        self.mod_linear_3    = torch.nn.Linear(10, 2)

    def forward(self, inputs):

        m = torch.nn.ELU()

        inputs     = m(self.mod_linear_1(inputs))
        inputs     = m(self.mod_linear_2(inputs))
        output     = m(self.mod_linear_3(inputs))

        output = self.power_constraint(output)

        return output

class demodulator(ENCBase):
    def __init__(self, args):
        super(demodulator, self).__init__(args)
        self.args             = args

        self.demod_linear_1    = torch.nn.Linear(2, 20)
        self.demod_linear_2    = torch.nn.Linear(20, 20)
        self.demod_linear_3    = torch.nn.Linear(20, 2)

    def forward(self, inputs):

        m = torch.nn.ELU()

        inputs     = m(self.demod_linear_1(inputs))
        inputs     = m(self.demod_linear_2(inputs))
        output     = m(self.demod_linear_3(inputs))

        return output

class ENC_interCNN_modified(ENCBase):
    def __init__(self, args):
        # turbofy only for code rate 1/3
        super(ENC_interCNN_modified, self).__init__(args)
        self.args             = args

        # Encoder
        if self.args.encoder == 'TurboAE_rate3_cnn':
            self.enc_cnn_1       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size
                                                      ,padding_mode=args.padding)

            self.enc_cnn_2       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size
                                                      ,padding_mode=args.padding)

            self.enc_cnn_3       = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size
                                                      ,padding_mode=args.padding)
        else: # Dense
            self.enc_cnn_1       = DenseSameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

            self.enc_cnn_2       = DenseSameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)

            self.enc_cnn_3       = DenseSameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k,
                                                      out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_2    = torch.nn.Linear(args.enc_num_unit, 1)
        self.enc_linear_3    = torch.nn.Linear(args.enc_num_unit, 1)



    def set_parallel(self):
        self.enc_cnn_1 = torch.nn.DataParallel(self.enc_cnn_1)
        self.enc_cnn_2 = torch.nn.DataParallel(self.enc_cnn_2)
        self.enc_cnn_3 = torch.nn.DataParallel(self.enc_cnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def forward(self, inputs, interleaver):

        inputs     = 2.0*inputs - 1.0

        x_sys_int  = interleaver(inputs)
        x_sys      = self.enc_cnn_1(x_sys_int)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        
        x_p2       = self.enc_cnn_3(inputs)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        codes = self.power_constraint(x_tx)

        return codes


class ENC_NNLTEencode(ENCBase):
    def __init__(self, args):
        # turbofy only for code rate 1/3
        super(ENC_NNLTEencode, self).__init__(args)
        self.args = args
        self.enc_cnn = SameShapeConv1d(num_layer=args.enc_num_layer, in_channels=args.code_rate_k*3,
                                       out_channels= args.enc_num_unit, kernel_size = args.enc_kernel_size,
                                       padding_mode=args.padding)
        self.enc_linear = torch.nn.Linear(args.enc_num_unit, 3)
        #self.interleaver = Interleaver(args, p_array)


    # def set_interleaver(self, p_array):
    #     self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_cnn = torch.nn.DataParallel(self.enc_cnn)
        self.enc_linear = torch.nn.DataParallel(self.enc_linear)

    def turbolte_encode(self, inputs):
        # Turbo LTE encode, y[n] = y[n-2] _+ y[n-3] _+ x[n] _+ x[n-1] _+ x[n-3]
        #                   Yihan: y[n] = f(y[n-1], ..., y[n-M]; x[n], ..., x[n-K])
        # TurboLTE + CNN + AWGN + DeepTurbo-RNN
        #      AWGN: L=100 (success), L=1000 (ongoing), L=40 (try it)
        #      Fading:
        #      Interference:
        #      TI on short block length.
        #
        # thinking about TurboLTE + CNN + Channel + CNN + TurboNet
        # NVDA: creating a library for Comm: TF function.

        input_shape = inputs.shape
        X_lte_inputs = torch.zeros((input_shape[0], input_shape[1], 1)).to(self.this_device)

        for idx in range(input_shape[1]):
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

    def forward(self, inputs, interleaver):

        inputs_1 = self.turbolte_encode(inputs)
        inputs_2 = self.turbolte_encode(interleaver(inputs))
        lte_codes = 2.0 * torch.cat([inputs, inputs_1, inputs_2], dim=2) - 1.0

        x_codes = self.enc_cnn(lte_codes)
        x_codes = self.enc_linear(x_codes)
        codes = self.power_constraint(x_codes)
        return codes

