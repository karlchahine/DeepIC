from interleavers import Interleaver, DeInterleaver

import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np

##################################################
# DTA Decoder with rate 1/3
# RNN version
##################################################

##################################################
# DTA Decoder with rate 1/3
# 1D CNN same shape decoder
##################################################

from encoders import SameShapeConv1d, DenseSameShapeConv1d
class DEC_LargeCNN(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DEC_LargeCNN, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        if self.args.encoder == 'TurboAE_rate3_cnn':
            CNNLayer = SameShapeConv1d
        else:
            CNNLayer = DenseSameShapeConv1d


        for idx in range(args.num_iteration):
            self.dec1_cnns.append(CNNLayer(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size
                                                  ,padding_mode=args.padding)
            )

            self.dec2_cnns.append(CNNLayer(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size
                                                  ,padding_mode=args.padding)
            )
            self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_cnns[idx] = torch.nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = torch.nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def forward(self, received, interleaver, deinterleaver):

        
        block_len = self.args.block_len

        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, block_len, 1))
        r_sys_int = interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            if self.args.interleaver == 'trainable':
                prior = torch.matmul(interleaver.perm_matrix.weight.t(), x_plr)
                
            else:
                prior      = deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.interleaver == 'trainable':
            final      = torch.sigmoid(torch.matmul(interleaver.perm_matrix.weight.t(), x_plr))

        else:
            final      = torch.sigmoid(deinterleaver(x_plr))

        return final



class DEC_LargeCNN_modified(torch.nn.Module):
    def __init__(self, args):
        super(DEC_LargeCNN_modified, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        if self.args.encoder == 'TurboAE_rate3_cnn':
            CNNLayer = SameShapeConv1d
        else:
            CNNLayer = DenseSameShapeConv1d


        for idx in range(args.num_iteration):
            self.dec1_cnns.append(CNNLayer(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size
                                                  ,padding_mode=args.padding)
            )

            self.dec2_cnns.append(CNNLayer(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.dec_num_unit, kernel_size = args.dec_kernel_size
                                                  ,padding_mode=args.padding)
            )
            self.dec1_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(args.dec_num_unit, args.num_iter_ft))

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_cnns[idx] = torch.nn.DataParallel(self.dec1_cnns[idx])
            self.dec2_cnns[idx] = torch.nn.DataParallel(self.dec2_cnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def forward(self, received, interleaver, deinterleaver):

        
        block_len = self.args.block_len

        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,2].view((self.args.batch_size, block_len, 1))
        r_sys_int = interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, block_len, 1))
        r_par2    = received[:,:,0].view((self.args.batch_size, block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            if self.args.interleaver == 'trainable':
                prior = torch.matmul(interleaver.perm_matrix.weight.t(), x_plr)
                
            else:
                prior      = deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.interleaver == 'trainable':
            final      = torch.sigmoid(torch.matmul(interleaver.perm_matrix.weight.t(), x_plr))

        else:
            final      = torch.sigmoid(deinterleaver(x_plr))

        return final




class DEC_LargeRNN(torch.nn.Module):
    def __init__(self, args):
        super(DEC_LargeRNN, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")



        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        self.dec1_rnns      = torch.nn.ModuleList()
        self.dec2_rnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                                        num_layers=2, bias=True, batch_first=True,
                                                        dropout=args.dropout, bidirectional=True)
            )

            self.dec2_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=args.dropout, bidirectional=True)
            )

            self.dec1_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return inputs

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_rnns[idx] = torch.nn.DataParallel(self.dec1_rnns[idx])
            self.dec2_rnns[idx] = torch.nn.DataParallel(self.dec2_rnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])



    def forward(self, received, interleaver, deinterleaver):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((self.args.batch_size, self.args.block_len, 1))
        r_sys_int = interleaver(r_sys)
        r_par1    = received[:,:,1].view((self.args.batch_size, self.args.block_len, 1))
        r_par2    = received[:,:,2].view((self.args.batch_size, self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            x_dec, _   = self.dec1_rnns[idx](x_this_dec)
            x_plr      = self.dec_act(self.dropout(self.dec1_outputs[idx](x_dec)))

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec, _   = self.dec2_rnns[idx](x_this_dec)
            x_plr      = self.dec_act(self.dropout(self.dec2_outputs[idx](x_dec)))

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)



        x_dec, _   = self.dec1_rnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec_act(self.dropout(self.dec1_outputs[self.args.num_iteration - 1](x_dec)))

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)


        x_dec, _   = self.dec2_rnns[self.args.num_iteration - 1](x_this_dec)

        x_plr      = self.dec_act(self.dropout(self.dec2_outputs[self.args.num_iteration - 1](x_dec)))

        logit      = deinterleaver(x_plr)

        final      = torch.sigmoid(logit)

        return final






