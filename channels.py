__author__ = 'yihanjiang'

import torch
from utils import snr_db2sigma, snr_sigma2db
import numpy as np

def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0, mode = 'encoder'):
    # SNRs at training
    if test_sigma == 'default':
        
        this_sigma_low = snr_db2sigma(snr_low)
        this_sigma_high= snr_db2sigma(snr_high)
        # mixture of noise sigma.
        this_sigma = (this_sigma_low - this_sigma_high) * torch.rand(noise_shape) + this_sigma_high

    else:

        this_sigma = snr_db2sigma(test_sigma)

    # SNRs at testing
    
    fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise



