__author__ = 'yihanjiang'

import torch
import torch.nn.functional as F

class Trainable_Interleaver(torch.nn.Module):
    def __init__(self, args):
        super(Trainable_Interleaver, self).__init__()
        self.args = args
        self.perm_matrix = torch.nn.Linear(self.args.block_len,self.args.block_len, bias=False)

    def forward(self, inputs):

        inputs = torch.transpose(inputs, 1,2)
        res    = self.perm_matrix(inputs)
        res = torch.transpose(res, 1,2)

        return res

        

class Interleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def forward(self, inputs):

        inputs = inputs.permute(1,0,2)
        res    = inputs[self.p_array]
        res    = res.permute(1, 0, 2)

        return res


class DeInterleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def set_parray(self, p_array):

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2)
        res    = inputs[self.reverse_p_array]
        res    = res.permute(1, 0, 2)

        return res
