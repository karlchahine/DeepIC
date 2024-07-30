__author__ = 'yihanjiang'
import torch
import time
import torch.nn.functional as F
from torch.autograd import Variable
from torch import linalg as LA

eps  = 1e-6

from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler
from loss import customized_loss
from channels import generate_noise

import numpy as np
from numpy import arange
from numpy.random import mtrand

######################################################################################
#
# Trainer, validation, and test for AE code design
#
######################################################################################

############# 2 USERS #################

def train_with_trainable_intrlvr(a,b,epoch, model, optimizer,intrlv_optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()
    alpha=a
    beta=b
    start_time = time.time()
    train_loss = 0.0
    train_loss_u1 = 0.0
    train_loss_u2 = 0.0

    perm_matrix_loss = 0.0
    perm_matrix_loss_u1 = 0.0
    perm_matrix_loss_u2 = 0.0

    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        X_train_u1    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u2    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if mode == 'encoder':
            fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
            fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2 = X_train_u1.to(device),X_train_u2.to(device), fwd_noise_u1.to(device), fwd_noise_u2.to(device)

        decoded1, decoded2= model(X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2)
        decoded1 = torch.clamp(decoded1, 0.0, 1.0)
        decoded2 = torch.clamp(decoded2, 0.0, 1.0)

        
        loss1=customized_loss(decoded1,X_train_u1,args, noise=fwd_noise_u1)
        rows_l1 = LA.norm(model.interleaver_u1.perm_matrix.weight,ord=1, axis=1)
        rows_l2 = LA.norm(model.interleaver_u1.perm_matrix.weight,ord=2, axis=1)

        cols_l1 = LA.norm(model.interleaver_u1.perm_matrix.weight,ord=1, axis=0)
        cols_l2 = LA.norm(model.interleaver_u1.perm_matrix.weight,ord=2, axis=0)

        rows_l1,rows_l2,cols_l1,cols_l2 = rows_l1.to(device), rows_l2.to(device),cols_l1.to(device),cols_l2.to(device)
        loss1_intrlvr = torch.sum(rows_l1) + torch.sum(cols_l1) - torch.sum(rows_l2) - torch.sum(cols_l2)
        loss1 = loss1 + 0.002*loss1_intrlvr 


        loss2=customized_loss(decoded2,X_train_u2,args, noise=fwd_noise_u2)
        rows_l1 = LA.norm(model.interleaver_u2.perm_matrix.weight,ord=1, axis=1)
        rows_l2 = LA.norm(model.interleaver_u2.perm_matrix.weight,ord=2, axis=1)

        cols_l1 = LA.norm(model.interleaver_u2.perm_matrix.weight,ord=1, axis=0)
        cols_l2 = LA.norm(model.interleaver_u2.perm_matrix.weight,ord=2, axis=0)

        rows_l1,rows_l2,cols_l1,cols_l2 = rows_l1.to(device), rows_l2.to(device),cols_l1.to(device),cols_l2.to(device)
        loss2_intrlvr = torch.sum(rows_l1) + torch.sum(cols_l1) - torch.sum(rows_l2) - torch.sum(cols_l2)
        loss2 = loss2 + 0.002*loss2_intrlvr 


        loss=alpha*loss1+beta*loss2

        optimizer.zero_grad()
        intrlv_optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        intrlv_optimizer.step()

        train_loss += loss.item()
        train_loss_u1 += loss1.item()
        train_loss_u2 += loss2.item()
                       


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    train_loss_u1 = train_loss_u1/(args.num_block/args.batch_size)
    train_loss_u2 = train_loss_u2/(args.num_block/args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

        

    return train_loss,train_loss_u1,train_loss_u2



def train(a,b,epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder', puncture=False):
   
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()
    alpha=a
    beta=b
    start_time = time.time()
    train_loss = 0.0
    train_loss_u1 = 0.0
    train_loss_u2 = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):


        if args.is_variable_block_len:
            block_len = np.random.randint(args.block_len_low, args.block_len_high)
        else:
            block_len = args.block_len

        

        
        X_train_u1    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u2    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if args.conditions == 'same':
            if mode == 'encoder':
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            else:
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        else:
            if mode == 'encoder':
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low+2, snr_high=args.train_enc_channel_high+2, mode = 'encoder')
            else:
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low+2, snr_high=args.train_dec_channel_high+2, mode = 'decoder')

        X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2 = X_train_u1.to(device),X_train_u2.to(device), fwd_noise_u1.to(device), fwd_noise_u2.to(device)

        decoded1, decoded2= model(X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2)
        decoded1 = torch.clamp(decoded1, 0.0, 1.0)
        decoded2 = torch.clamp(decoded2, 0.0, 1.0)

        
        loss1=customized_loss(decoded1,X_train_u1,args, noise=fwd_noise_u1)
        loss2=customized_loss(decoded2,X_train_u2,args, noise=fwd_noise_u2)

        loss=alpha*loss1+beta*loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_u1 += loss1.item()
        train_loss_u2 += loss2.item()
                       


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    train_loss_u1 = train_loss_u1/(args.num_block/args.batch_size)
    train_loss_u2 = train_loss_u2/(args.num_block/args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

        

    return train_loss,train_loss_u1,train_loss_u2



def validate(model, args, use_cuda = False, verbose = True, puncture=False):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss_u1, test_custom_loss_u1, test_ber_u1= 0.0, 0.0, 0.0
    test_bce_loss_u2, test_custom_loss_u2, test_ber_u2= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            if args.conditions == 'same':
                fwd_noise_u1  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)
                fwd_noise_u2  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)

            else:
                fwd_noise_u1  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)
                fwd_noise_u2  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low+2,
                                            snr_high=args.train_enc_channel_low+2)

            X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)


            decoded1, decoded2= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
            
            decoded1 = torch.clamp(decoded1, 0.0, 1.0)
            decoded2 = torch.clamp(decoded2, 0.0, 1.0)

            decoded1 = decoded1.detach()
            decoded2 = decoded2.detach()

            X_test_u1 = X_test_u1.detach()
            X_test_u2 = X_test_u2.detach()

            ##User 1
            test_bce_loss_u1 += F.binary_cross_entropy(decoded1, X_test_u1)
            test_custom_loss_u1 += customized_loss(decoded1, X_test_u1, noise = fwd_noise_u1, args = args)
            test_ber_u1  += errors_ber(decoded1, X_test_u1)

            ##User 2
            test_bce_loss_u2 += F.binary_cross_entropy(decoded2, X_test_u2)
            test_custom_loss_u2 += customized_loss(decoded2, X_test_u2, noise = fwd_noise_u2, args = args)
            test_ber_u2  += errors_ber(decoded2, X_test_u2)

    test_bce_loss_u1 /= num_test_batch
    test_custom_loss_u1 /= num_test_batch
    test_ber_u1  /= num_test_batch

    test_bce_loss_u2 /= num_test_batch
    test_custom_loss_u2 /= num_test_batch
    test_ber_u2  /= num_test_batch

    if verbose:
        print('====> User1: Test set BCE loss', float(test_bce_loss_u1),
              'Custom Loss',float(test_custom_loss_u1),
              'with ber ', float(test_ber_u1),
        )

        print('====> User2: Test set BCE loss', float(test_bce_loss_u2),
              'Custom Loss',float(test_custom_loss_u2),
              'with ber ', float(test_ber_u2),
        )

    # report_loss = float(test_bce_loss)
    # report_ber  = float(test_ber)

    # return report_loss, report_ber


def test(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    
    block_len = args.block_len
    

    ber_res_u1, ber_res_u2, bler_res_u1,bler_res_u2 = [], [], [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    #snrs = [-1.5,0,1.5,3,6,9,12,15,18,21,24]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber_u1,test_ber_u2, test_bler_u1,test_bler_u2 = .0, .0, .0, .0
        with torch.no_grad():
            num_test_batch = int(5*args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)

                
                decoded1, decoded2= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
                

                ##User 1
                test_ber_u1  += errors_ber(decoded1, X_test_u1)
                test_bler_u1 += errors_bler(decoded1,X_test_u1)

                ##User 2
                test_ber_u2  += errors_ber(decoded2, X_test_u2)
                test_bler_u2 += errors_bler(decoded2,X_test_u2)


        test_ber_u1  /= num_test_batch
        test_bler_u1 /= num_test_batch

        test_ber_u2  /= num_test_batch
        test_bler_u2 /= num_test_batch       
        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber_u1), 'with bler', float(test_bler_u1))
        print('User2: Test SNR',this_snr ,'with ber ', float(test_ber_u2), 'with bler', float(test_bler_u2))
        ber_res_u1.append(float(test_ber_u1))
        bler_res_u1.append( float(test_bler_u1))

        ber_res_u2.append(float(test_ber_u2))
        bler_res_u2.append( float(test_bler_u2))


    print('User 1 final results on SNRs ', snrs)
    print('BER', ber_res_u1)
    print('BLER', bler_res_u1)

    print('User 2 final results on SNRs ', snrs)
    print('BER', ber_res_u2)
    print('BLER', bler_res_u2)


def test_ber_per_pos(model, args, snr):
    
    device = torch.device("cuda") 
    model.eval()

    ber_res_u1 = torch.zeros(args.block_len,).to(device)
    ber_res_u2 = torch.zeros(args.block_len,).to(device)

    with torch.no_grad():
        #num_test_batch = int(5*args.num_block/(args.batch_size))
        num_test_batch = 2000
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=snr)
            fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=snr)

            X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)

            decoded1, decoded2= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)

            
            ##User 1
            ber_res_u1  += errors_ber_pos(decoded1, X_test_u1)

            ##User 2
            ber_res_u2  += errors_ber_pos(decoded2, X_test_u2)

    ber_res_u1  /= num_test_batch
    ber_res_u2 /= num_test_batch
   
    return ber_res_u1, ber_res_u2

def calc_statistics(model, args):
    
    device = torch.device("cuda") 
    model.eval()

    mean = torch.zeros(1,).to(device)
    var = torch.zeros(1,).to(device)

    with torch.no_grad():
        #num_test_batch = int(5*args.num_block/(args.batch_size))
        num_test_batch = 2000
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

            X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)
            enc1 = model.encoder1(X_test_u1)
            
   
    return ber_res_u1, ber_res_u2

def test_0_middle(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    
    block_len = args.block_len
    

    ber_res_u1, ber_res_u2, bler_res_u1,bler_res_u2 = [], [], [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    #snrs = [-1.5,0,1.5,3,6,9,12,15,18,21,24]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber_u1,test_ber_u2, test_bler_u1,test_bler_u2 = .0, .0, .0, .0
        with torch.no_grad():
            num_test_batch = int(5*args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)
                
                decoded1, decoded2= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
                

                ##User 1
                test_ber_u1  += errors_ber(decoded1, X_test_u1)
                test_bler_u1 += errors_bler(decoded1,X_test_u1)

                ##User 2
                test_ber_u2  += errors_ber(decoded2, X_test_u2)
                test_bler_u2 += errors_bler(decoded2,X_test_u2)


        test_ber_u1  /= num_test_batch
        test_bler_u1 /= num_test_batch

        test_ber_u2  /= num_test_batch
        test_bler_u2 /= num_test_batch       
        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber_u1), 'with bler', float(test_bler_u1))
        print('User2: Test SNR',this_snr ,'with ber ', float(test_ber_u2), 'with bler', float(test_bler_u2))
        ber_res_u1.append(float(test_ber_u1))
        bler_res_u1.append( float(test_bler_u1))

        ber_res_u2.append(float(test_ber_u2))
        bler_res_u2.append( float(test_bler_u2))


    print('User 1 final results on SNRs ', snrs)
    print('BER', ber_res_u1)
    print('BLER', bler_res_u1)

    print('User 2 final results on SNRs ', snrs)
    print('BER', ber_res_u2)
    print('BLER', bler_res_u2)



############# 1 USER #################

def train_with_trainable_intrlvr_1user(epoch, model, optimizer,intrlv_optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()

    start_time = time.time()
    train_loss = 0.0

    perm_matrix_loss = 0.0

    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if mode == 'encoder':
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

        decoded = model(X_train, fwd_noise)
        decoded = torch.clamp(decoded, 0.0, 1.0)

        
        loss=customized_loss(decoded,X_train,args, noise=fwd_noise)
        rows_l1 = LA.norm(model.interleaver.perm_matrix.weight,ord=1, axis=1)
        rows_l2 = LA.norm(model.interleaver.perm_matrix.weight,ord=2, axis=1)

        cols_l1 = LA.norm(model.interleaver.perm_matrix.weight,ord=1, axis=0)
        cols_l2 = LA.norm(model.interleaver.perm_matrix.weight,ord=2, axis=0)

        rows_l1,rows_l2,cols_l1,cols_l2 = rows_l1.to(device), rows_l2.to(device),cols_l1.to(device),cols_l2.to(device)
        loss_intrlvr = torch.sum(rows_l1) + torch.sum(cols_l1) - torch.sum(rows_l2) - torch.sum(cols_l2)
        
        loss = loss + 0.002*loss_intrlvr 


        optimizer.zero_grad()
        intrlv_optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        intrlv_optimizer.step()

        train_loss += loss.item()


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
 
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss



def train_1user(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):
   
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()

    start_time = time.time()
    train_loss = 0.0

    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if mode == 'encoder':
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

        decoded= model(X_train, fwd_noise)
        decoded = torch.clamp(decoded, 0.0, 1.0)

        
        loss=customized_loss(decoded,X_train,args, noise=fwd_noise)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
 

    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
 
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss





def test_1user(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    bers = []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber = .0
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

                decoded = model(X_test, fwd_noise)
                
                ##User 1
                test_ber  += errors_ber(decoded, X_test)


        test_ber  /= num_test_batch

    
        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber))
        bers.append(float(test_ber))


    print('User 1 final results on SNRs ', snrs)
    print('BER', bers)



def validate_1user(model, args, use_cuda = False, verbose = True):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss, test_ber= 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            fwd_noise  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)


            X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)


            decoded= model(X_test, fwd_noise)
            
            decoded = torch.clamp(decoded, 0.0, 1.0)

            decoded = decoded.detach()

            X_test = X_test.detach()

            ##User 1
            test_bce_loss += F.binary_cross_entropy(decoded, X_test)
            test_ber  += errors_ber(decoded, X_test)



    test_bce_loss /= num_test_batch
    test_ber  /= num_test_batch


    if verbose:
        print('====> User1: Test set BCE loss', float(test_bce_loss),
              'with ber ', float(test_ber),
        )




############# 3 USERS #################

def train_3users(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):
   
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()

    start_time = time.time()
    train_loss = 0.0
    train_loss_u1 = 0.0
    train_loss_u2 = 0.0
    train_loss_u3 = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        X_train_u1    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u2    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u3    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if mode == 'encoder':
            fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            fwd_noise_u3  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
            fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
            fwd_noise_u3  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train_u1,X_train_u2,X_train_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3 = X_train_u1.to(device),X_train_u2.to(device), X_train_u3.to(device), fwd_noise_u1.to(device), fwd_noise_u2.to(device), fwd_noise_u3.to(device)

        decoded1, decoded2, decoded3 = model(X_train_u1,X_train_u2,X_train_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3)

        decoded1 = torch.clamp(decoded1, 0.0, 1.0)
        decoded2 = torch.clamp(decoded2, 0.0, 1.0)
        decoded3 = torch.clamp(decoded3, 0.0, 1.0)

        
        loss1=customized_loss(decoded1,X_train_u1,args, noise=fwd_noise_u1)
        loss2=customized_loss(decoded2,X_train_u2,args, noise=fwd_noise_u2)
        loss3=customized_loss(decoded3,X_train_u3,args, noise=fwd_noise_u3)

        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_u1 += loss1.item()
        train_loss_u2 += loss2.item()
        train_loss_u3 += loss3.item()
                       


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    train_loss_u1 = train_loss_u1/(args.num_block/args.batch_size)
    train_loss_u2 = train_loss_u2/(args.num_block/args.batch_size)
    train_loss_u3 = train_loss_u3/(args.num_block/args.batch_size)

    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss,train_loss_u1,train_loss_u2,train_loss_u3





def test_3users(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    
    ber_res_u1, ber_res_u2, ber_res_u3, bler_res_u1,bler_res_u2,bler_res_u3 = [], [], [], [],[], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber_u1,test_ber_u2,test_ber_u3, test_bler_u1,test_bler_u2,test_bler_u3 = .0, .0, .0, .0,.0, .0
        with torch.no_grad():
            num_test_batch = int(5*args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u3     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u3  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test_u1, X_test_u2,X_test_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3= X_test_u1.to(device),X_test_u2.to(device),X_test_u3.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device),fwd_noise_u3.to(device)

                
                decoded1, decoded2, decoded3= model(X_test_u1,X_test_u2,X_test_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3)
                

                ##User 1
                test_ber_u1  += errors_ber(decoded1, X_test_u1)
                test_bler_u1 += errors_bler(decoded1,X_test_u1)

                ##User 2
                test_ber_u2  += errors_ber(decoded2, X_test_u2)
                test_bler_u2 += errors_bler(decoded2,X_test_u2)

                ##User 2
                test_ber_u3  += errors_ber(decoded3, X_test_u3)
                test_bler_u3 += errors_bler(decoded3,X_test_u3)


        test_ber_u1  /= num_test_batch
        test_bler_u1 /= num_test_batch

        test_ber_u2  /= num_test_batch
        test_bler_u2 /= num_test_batch

        test_ber_u3  /= num_test_batch
        test_bler_u3 /= num_test_batch 

        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber_u1), 'with bler', float(test_bler_u1))
        print('User2: Test SNR',this_snr ,'with ber ', float(test_ber_u2), 'with bler', float(test_bler_u2))
        print('User3: Test SNR',this_snr ,'with ber ', float(test_ber_u3), 'with bler', float(test_bler_u3))

        ber_res_u1.append(float(test_ber_u1))
        bler_res_u1.append( float(test_bler_u1))

        ber_res_u2.append(float(test_ber_u2))
        bler_res_u2.append( float(test_bler_u2))

        ber_res_u3.append(float(test_ber_u3))
        bler_res_u3.append( float(test_bler_u3))


    print('User 1 final results on SNRs ', snrs)
    print('BER', ber_res_u1)
    print('BLER', bler_res_u1)

    print('User 2 final results on SNRs ', snrs)
    print('BER', ber_res_u2)
    print('BLER', bler_res_u2)

    print('User 3 final results on SNRs ', snrs)
    print('BER', ber_res_u3)
    print('BLER', bler_res_u3)


def validate_3users(model, args, use_cuda = False, verbose = True):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss_u1, test_custom_loss_u1, test_ber_u1= 0.0, 0.0, 0.0
    test_bce_loss_u2, test_custom_loss_u2, test_ber_u2= 0.0, 0.0, 0.0
    test_bce_loss_u3, test_custom_loss_u3, test_ber_u3= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u3     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            fwd_noise_u1  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)
            fwd_noise_u2  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)
            fwd_noise_u3  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)

            X_test_u1, X_test_u2, X_test_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3= X_test_u1.to(device),X_test_u2.to(device),X_test_u3.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device),fwd_noise_u3.to(device)


            decoded1, decoded2, decoded3= model(X_test_u1,X_test_u2,X_test_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3)
            
            decoded1 = torch.clamp(decoded1, 0.0, 1.0)
            decoded2 = torch.clamp(decoded2, 0.0, 1.0)
            decoded3 = torch.clamp(decoded3, 0.0, 1.0)

            decoded1 = decoded1.detach()
            decoded2 = decoded2.detach()
            decoded3 = decoded3.detach()

            X_test_u1 = X_test_u1.detach()
            X_test_u2 = X_test_u2.detach()
            X_test_u3 = X_test_u3.detach()

            ##User 1
            test_bce_loss_u1 += F.binary_cross_entropy(decoded1, X_test_u1)
            test_custom_loss_u1 += customized_loss(decoded1, X_test_u1, noise = fwd_noise_u1, args = args)
            test_ber_u1  += errors_ber(decoded1, X_test_u1)

            ##User 2
            test_bce_loss_u2 += F.binary_cross_entropy(decoded2, X_test_u2)
            test_custom_loss_u2 += customized_loss(decoded2, X_test_u2, noise = fwd_noise_u2, args = args)
            test_ber_u2  += errors_ber(decoded2, X_test_u2)

            ##User 3
            test_bce_loss_u3 += F.binary_cross_entropy(decoded3, X_test_u3)
            test_custom_loss_u3 += customized_loss(decoded3, X_test_u3, noise = fwd_noise_u3, args = args)
            test_ber_u3  += errors_ber(decoded3, X_test_u3)

    test_bce_loss_u1 /= num_test_batch
    test_custom_loss_u1 /= num_test_batch
    test_ber_u1  /= num_test_batch

    test_bce_loss_u2 /= num_test_batch
    test_custom_loss_u2 /= num_test_batch
    test_ber_u2  /= num_test_batch

    test_bce_loss_u3 /= num_test_batch
    test_custom_loss_u3 /= num_test_batch
    test_ber_u3  /= num_test_batch

    if verbose:
        print('====> User1: Test set BCE loss', float(test_bce_loss_u1),
              'with ber ', float(test_ber_u1),
        )

        print('====> User2: Test set BCE loss', float(test_bce_loss_u2),
              'with ber ', float(test_ber_u2),
        )

        print('====> User3: Test set BCE loss', float(test_bce_loss_u3),
              'with ber ', float(test_ber_u3),
        )



############# 4 USERS #################

def train_4users(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):
   
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()

    start_time = time.time()
    train_loss = 0.0
    train_loss_u1 = 0.0
    train_loss_u2 = 0.0
    train_loss_u3 = 0.0
    train_loss_u4 = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        X_train_u1    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u2    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u3    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u4    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if mode == 'encoder':
            fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            fwd_noise_u3  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            fwd_noise_u4  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
            fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
            fwd_noise_u3  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
            fwd_noise_u4  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train_u1,X_train_u2,X_train_u3,X_train_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3,fwd_noise_u4 = X_train_u1.to(device),X_train_u2.to(device), X_train_u3.to(device), X_train_u4.to(device), fwd_noise_u1.to(device), fwd_noise_u2.to(device), fwd_noise_u3.to(device), fwd_noise_u4.to(device)

        decoded1, decoded2, decoded3, decoded4 = model(X_train_u1,X_train_u2,X_train_u3,X_train_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3,fwd_noise_u4)

        decoded1 = torch.clamp(decoded1, 0.0, 1.0)
        decoded2 = torch.clamp(decoded2, 0.0, 1.0)
        decoded3 = torch.clamp(decoded3, 0.0, 1.0)
        decoded4 = torch.clamp(decoded4, 0.0, 1.0)

        
        loss1=customized_loss(decoded1,X_train_u1,args, noise=fwd_noise_u1)
        loss2=customized_loss(decoded2,X_train_u2,args, noise=fwd_noise_u2)
        loss3=customized_loss(decoded3,X_train_u3,args, noise=fwd_noise_u3)
        loss4=customized_loss(decoded4,X_train_u4,args, noise=fwd_noise_u4)

        loss = loss1 + loss2 + loss3 + loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_u1 += loss1.item()
        train_loss_u2 += loss2.item()
        train_loss_u3 += loss3.item()
        train_loss_u4 += loss4.item()
                       


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    train_loss_u1 = train_loss_u1/(args.num_block/args.batch_size)
    train_loss_u2 = train_loss_u2/(args.num_block/args.batch_size)
    train_loss_u3 = train_loss_u3/(args.num_block/args.batch_size)
    train_loss_u4 = train_loss_u4/(args.num_block/args.batch_size)

    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss,train_loss_u1,train_loss_u2,train_loss_u3,train_loss_u4





def test_4users(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    
    ber_res_u1, ber_res_u2, ber_res_u3, ber_res_u4, bler_res_u1,bler_res_u2,bler_res_u3,bler_res_u4 = [], [], [], [],[], [],[],[]
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber_u1,test_ber_u2,test_ber_u3,test_ber_u4, test_bler_u1,test_bler_u2,test_bler_u3,test_bler_u4 = .0, .0, .0, .0,.0, .0, .0, .0
        with torch.no_grad():
            num_test_batch = int(5*args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u3     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u4     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u3  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u4  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test_u1, X_test_u2,X_test_u3,X_test_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3,fwd_noise_u4= X_test_u1.to(device),X_test_u2.to(device),X_test_u3.to(device),X_test_u4.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device),fwd_noise_u3.to(device),fwd_noise_u4.to(device)

                
                decoded1, decoded2, decoded3, decoded4= model(X_test_u1,X_test_u2,X_test_u3,X_test_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3,fwd_noise_u4)
                

                ##User 1
                test_ber_u1  += errors_ber(decoded1, X_test_u1)
                test_bler_u1 += errors_bler(decoded1,X_test_u1)

                ##User 2
                test_ber_u2  += errors_ber(decoded2, X_test_u2)
                test_bler_u2 += errors_bler(decoded2,X_test_u2)

                ##User 3
                test_ber_u3  += errors_ber(decoded3, X_test_u3)
                test_bler_u3 += errors_bler(decoded3,X_test_u3)

                ##User 4
                test_ber_u4  += errors_ber(decoded4, X_test_u4)
                test_bler_u4 += errors_bler(decoded4,X_test_u4)


        test_ber_u1  /= num_test_batch
        test_bler_u1 /= num_test_batch

        test_ber_u2  /= num_test_batch
        test_bler_u2 /= num_test_batch

        test_ber_u3  /= num_test_batch
        test_bler_u3 /= num_test_batch

        test_ber_u4  /= num_test_batch
        test_bler_u4 /= num_test_batch 

        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber_u1), 'with bler', float(test_bler_u1))
        print('User2: Test SNR',this_snr ,'with ber ', float(test_ber_u2), 'with bler', float(test_bler_u2))
        print('User3: Test SNR',this_snr ,'with ber ', float(test_ber_u3), 'with bler', float(test_bler_u3))
        print('User4: Test SNR',this_snr ,'with ber ', float(test_ber_u4), 'with bler', float(test_bler_u4))

        ber_res_u1.append(float(test_ber_u1))
        bler_res_u1.append( float(test_bler_u1))

        ber_res_u2.append(float(test_ber_u2))
        bler_res_u2.append( float(test_bler_u2))

        ber_res_u3.append(float(test_ber_u3))
        bler_res_u3.append( float(test_bler_u3))

        ber_res_u4.append(float(test_ber_u4))
        bler_res_u4.append( float(test_bler_u4))


    print('User 1 final results on SNRs ', snrs)
    print('BER', ber_res_u1)
    print('BLER', bler_res_u1)

    print('User 2 final results on SNRs ', snrs)
    print('BER', ber_res_u2)
    print('BLER', bler_res_u2)

    print('User 3 final results on SNRs ', snrs)
    print('BER', ber_res_u3)
    print('BLER', bler_res_u3)

    print('User 4 final results on SNRs ', snrs)
    print('BER', ber_res_u4)
    print('BLER', bler_res_u4)


def validate_4users(model, args, use_cuda = False, verbose = True):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss_u1, test_custom_loss_u1, test_ber_u1= 0.0, 0.0, 0.0
    test_bce_loss_u2, test_custom_loss_u2, test_ber_u2= 0.0, 0.0, 0.0
    test_bce_loss_u3, test_custom_loss_u3, test_ber_u3= 0.0, 0.0, 0.0
    test_bce_loss_u4, test_custom_loss_u4, test_ber_u4= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u3     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u4     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            fwd_noise_u1  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)
            fwd_noise_u2  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)
            fwd_noise_u3  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)
            fwd_noise_u4  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)                           

            X_test_u1, X_test_u2, X_test_u3, X_test_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3, fwd_noise_u4= X_test_u1.to(device),X_test_u2.to(device),X_test_u3.to(device), X_test_u4.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device),fwd_noise_u3.to(device), fwd_noise_u4.to(device)


            decoded1, decoded2, decoded3, decoded4= model(X_test_u1,X_test_u2,X_test_u3,X_test_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3,fwd_noise_u4)
            
            decoded1 = torch.clamp(decoded1, 0.0, 1.0)
            decoded2 = torch.clamp(decoded2, 0.0, 1.0)
            decoded3 = torch.clamp(decoded3, 0.0, 1.0)
            decoded4 = torch.clamp(decoded4, 0.0, 1.0)

            decoded1 = decoded1.detach()
            decoded2 = decoded2.detach()
            decoded3 = decoded3.detach()
            decoded4 = decoded4.detach()

            X_test_u1 = X_test_u1.detach()
            X_test_u2 = X_test_u2.detach()
            X_test_u3 = X_test_u3.detach()
            X_test_u4 = X_test_u4.detach()

            ##User 1
            test_bce_loss_u1 += F.binary_cross_entropy(decoded1, X_test_u1)
            test_custom_loss_u1 += customized_loss(decoded1, X_test_u1, noise = fwd_noise_u1, args = args)
            test_ber_u1  += errors_ber(decoded1, X_test_u1)

            ##User 2
            test_bce_loss_u2 += F.binary_cross_entropy(decoded2, X_test_u2)
            test_custom_loss_u2 += customized_loss(decoded2, X_test_u2, noise = fwd_noise_u2, args = args)
            test_ber_u2  += errors_ber(decoded2, X_test_u2)

            ##User 3
            test_bce_loss_u3 += F.binary_cross_entropy(decoded3, X_test_u3)
            test_custom_loss_u3 += customized_loss(decoded3, X_test_u3, noise = fwd_noise_u3, args = args)
            test_ber_u3  += errors_ber(decoded3, X_test_u3)

            ##User 4
            test_bce_loss_u4 += F.binary_cross_entropy(decoded4, X_test_u4)
            test_custom_loss_u4 += customized_loss(decoded4, X_test_u4, noise = fwd_noise_u4, args = args)
            test_ber_u4  += errors_ber(decoded4, X_test_u4)

    test_bce_loss_u1 /= num_test_batch
    test_custom_loss_u1 /= num_test_batch
    test_ber_u1  /= num_test_batch

    test_bce_loss_u2 /= num_test_batch
    test_custom_loss_u2 /= num_test_batch
    test_ber_u2  /= num_test_batch

    test_bce_loss_u3 /= num_test_batch
    test_custom_loss_u3 /= num_test_batch
    test_ber_u3  /= num_test_batch

    test_bce_loss_u4 /= num_test_batch
    test_custom_loss_u4 /= num_test_batch
    test_ber_u4  /= num_test_batch

    if verbose:
        print('====> User1: Test set BCE loss', float(test_bce_loss_u1),
              'with ber ', float(test_ber_u1),
        )

        print('====> User2: Test set BCE loss', float(test_bce_loss_u2),
              'with ber ', float(test_ber_u2),
        )

        print('====> User3: Test set BCE loss', float(test_bce_loss_u3),
              'with ber ', float(test_ber_u3),
        )

        print('====> User4: Test set BCE loss', float(test_bce_loss_u4),
              'with ber ', float(test_ber_u4),
        )


####SIC stuff

def train_sic(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder', puncture=False):
       
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()

    start_time = time.time()
    train_loss = 0.0
    train_loss_u1 = 0.0
    train_loss_u2 = 0.0
    final_loss_u1 = 0.0
    final_loss_u2 = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):


        block_len = args.block_len

            
        X_train_u1    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u2    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if args.conditions == 'same':
            if mode == 'encoder':
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            else:
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        else:
            if mode == 'encoder':
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low+2, snr_high=args.train_enc_channel_high+2, mode = 'encoder')
            else:
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low+2, snr_high=args.train_dec_channel_high+2, mode = 'decoder')

        X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2 = X_train_u1.to(device),X_train_u2.to(device), fwd_noise_u1.to(device), fwd_noise_u2.to(device)

        decoded1, decoded2, finaldecoded1, finaldecoded2 = model(X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2)

        decoded1 = torch.clamp(decoded1, 0.0, 1.0)
        decoded2 = torch.clamp(decoded2, 0.0, 1.0)
        finaldecoded1 = torch.clamp(finaldecoded1, 0.0, 1.0)
        finaldecoded2 = torch.clamp(finaldecoded2, 0.0, 1.0)

        ##User 1 estimate msg 2
        loss1=customized_loss(decoded1,X_train_u2,args, noise=fwd_noise_u1)

        ##User 2 estimate msg 1
        loss2=customized_loss(decoded2,X_train_u1,args, noise=fwd_noise_u1)

        ##User 1 estimate msg 1
        loss3=customized_loss(finaldecoded1,X_train_u1,args, noise=fwd_noise_u1)

        ##User 2 estimate msg 2
        loss4=customized_loss(finaldecoded2,X_train_u2,args, noise=fwd_noise_u1)

        loss = loss1 + loss2 + loss3 + loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_u1 += loss1.item()
        train_loss_u2 += loss2.item()
        final_loss_u1 += loss3.item()
        final_loss_u2 += loss4.item()
                       


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    train_loss_u1 = train_loss_u1/(args.num_block/args.batch_size)
    train_loss_u2 = train_loss_u2/(args.num_block/args.batch_size)
    final_loss_u1 = final_loss_u1/(args.num_block/args.batch_size)
    final_loss_u2 = final_loss_u2/(args.num_block/args.batch_size)


    return train_loss,train_loss_u1,train_loss_u2,final_loss_u1, final_loss_u2


def validate_sic(model, args, use_cuda = False, verbose = True, puncture=False):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss_u1, test_custom_loss_u1, test_ber_u1= 0.0, 0.0, 0.0
    test_bce_loss_u2, test_custom_loss_u2, test_ber_u2= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            if args.conditions == 'same':
                fwd_noise_u1  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)
                fwd_noise_u2  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)

            else:
                fwd_noise_u1  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)
                fwd_noise_u2  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low+2,
                                            snr_high=args.train_enc_channel_low+2)

            X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)


            decoded1, decoded2, finaldecoded1, finaldecoded2= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
            
            decoded1 = torch.clamp(decoded1, 0.0, 1.0)
            decoded2 = torch.clamp(decoded2, 0.0, 1.0)
            finaldecoded1 = torch.clamp(finaldecoded1, 0.0, 1.0)
            finaldecoded2 = torch.clamp(finaldecoded2, 0.0, 1.0)

            decoded1 = decoded1.detach()
            decoded2 = decoded2.detach()
            finaldecoded1 = finaldecoded1.detach()
            finaldecoded2 = finaldecoded2.detach()

            X_test_u1 = X_test_u1.detach()
            X_test_u2 = X_test_u2.detach()

            ##User 1
            test_bce_loss_u1 += F.binary_cross_entropy(finaldecoded1, X_test_u1)
            test_custom_loss_u1 += customized_loss(finaldecoded1, X_test_u1, noise = fwd_noise_u1, args = args)
            test_ber_u1  += errors_ber(finaldecoded1, X_test_u1)

            ##User 2
            test_bce_loss_u2 += F.binary_cross_entropy(finaldecoded2, X_test_u2)
            test_custom_loss_u2 += customized_loss(finaldecoded2, X_test_u2, noise = fwd_noise_u2, args = args)
            test_ber_u2  += errors_ber(finaldecoded2, X_test_u2)

    test_bce_loss_u1 /= num_test_batch
    test_custom_loss_u1 /= num_test_batch
    test_ber_u1  /= num_test_batch

    test_bce_loss_u2 /= num_test_batch
    test_custom_loss_u2 /= num_test_batch
    test_ber_u2  /= num_test_batch

    if verbose:
        print('====> User1: Test set BCE loss', float(test_bce_loss_u1),
              'Custom Loss',float(test_custom_loss_u1),
              'with ber ', float(test_ber_u1),
        )

        print('====> User2: Test set BCE loss', float(test_bce_loss_u2),
              'Custom Loss',float(test_custom_loss_u2),
              'with ber ', float(test_ber_u2),
        )

    # report_loss = float(test_bce_loss)
    # report_ber  = float(test_ber)

    # return report_loss, report_ber


def test_sic(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    
    block_len = args.block_len
    

    ber_res_u1, ber_res_u2, bler_res_u1,bler_res_u2 = [], [], [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    #snrs = [-1.5,0,1.5,3,6,9,12,15,18,21,24]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber_u1,test_ber_u2, test_bler_u1,test_bler_u2 = .0, .0, .0, .0
        with torch.no_grad():
            num_test_batch = int(5*args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)

                
                decoded1, decoded2, finaldecoded1, finaldecoded2= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
                

                ##User 1
                test_ber_u1  += errors_ber(finaldecoded1, X_test_u1)
                test_bler_u1 += errors_bler(finaldecoded1,X_test_u1)

                ##User 2
                test_ber_u2  += errors_ber(finaldecoded2, X_test_u2)
                test_bler_u2 += errors_bler(finaldecoded2,X_test_u2)


        test_ber_u1  /= num_test_batch
        test_bler_u1 /= num_test_batch

        test_ber_u2  /= num_test_batch
        test_bler_u2 /= num_test_batch       
        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber_u1), 'with bler', float(test_bler_u1))
        print('User2: Test SNR',this_snr ,'with ber ', float(test_ber_u2), 'with bler', float(test_bler_u2))
        ber_res_u1.append(float(test_ber_u1))
        bler_res_u1.append( float(test_bler_u1))

        ber_res_u2.append(float(test_ber_u2))
        bler_res_u2.append( float(test_bler_u2))


    print('User 1 final results on SNRs ', snrs)
    print('BER', ber_res_u1)
    print('BLER', bler_res_u1)

    print('User 2 final results on SNRs ', snrs)
    print('BER', ber_res_u2)
    print('BLER', bler_res_u2)


####MTL SIC stuff
def train_sic_MTL(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder', puncture=False):
       
    device = torch.device("cuda" if use_cuda else "cpu")
    model.train()

    start_time = time.time()
    train_loss = 0.0
    train_loss_u1 = 0.0
    train_loss_u2 = 0.0
    final_loss_u1 = 0.0
    final_loss_u2 = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):


        block_len = args.block_len

            
        X_train_u1    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        X_train_u2    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

        if args.conditions == 'same':
            if mode == 'encoder':
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
            else:
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        else:
            if mode == 'encoder':
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low+2, snr_high=args.train_enc_channel_high+2, mode = 'encoder')
            else:
                fwd_noise_u1  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
                fwd_noise_u2  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low+2, snr_high=args.train_dec_channel_high+2, mode = 'decoder')

        X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2 = X_train_u1.to(device),X_train_u2.to(device), fwd_noise_u1.to(device), fwd_noise_u2.to(device)

        dec1_self, dec1_other, dec2_self, dec2_other = model(X_train_u1,X_train_u2, fwd_noise_u1,fwd_noise_u2)

        dec1_self = torch.clamp(dec1_self, 0.0, 1.0)
        dec1_other = torch.clamp(dec1_other, 0.0, 1.0)
        dec2_self = torch.clamp(dec2_self, 0.0, 1.0)
        dec2_other = torch.clamp(dec2_other, 0.0, 1.0)

        ##User 1 estimate msg 2
        loss1=customized_loss(dec1_other,X_train_u2,args, noise=fwd_noise_u1)

        ##User 2 estimate msg 1
        loss2=customized_loss(dec2_other,X_train_u1,args, noise=fwd_noise_u1)

        ##User 1 estimate msg 1
        loss3=customized_loss(dec1_self,X_train_u1,args, noise=fwd_noise_u1)

        ##User 2 estimate msg 2
        loss4=customized_loss(dec2_self,X_train_u2,args, noise=fwd_noise_u1)

        #loss = 0.2*loss1 + 0.2*loss2 + loss3 + loss4

        loss = loss3 + loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_u1 += loss1.item()
        train_loss_u2 += loss2.item()
        final_loss_u1 += loss3.item()
        final_loss_u2 += loss4.item()
                       


    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    train_loss_u1 = train_loss_u1/(args.num_block/args.batch_size)
    train_loss_u2 = train_loss_u2/(args.num_block/args.batch_size)
    final_loss_u1 = final_loss_u1/(args.num_block/args.batch_size)
    final_loss_u2 = final_loss_u2/(args.num_block/args.batch_size)


    return train_loss,train_loss_u1,train_loss_u2,final_loss_u1, final_loss_u2


def validate_sic_MTL(model, args, use_cuda = False, verbose = True, puncture=False):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss_u1, test_custom_loss_u1, test_ber_u1= 0.0, 0.0, 0.0
    test_bce_loss_u2, test_custom_loss_u2, test_ber_u2= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

            if args.conditions == 'same':
                fwd_noise_u1  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)
                fwd_noise_u2  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)

            else:
                fwd_noise_u1  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low,
                                            snr_high=args.train_enc_channel_low)
                fwd_noise_u2  = generate_noise(noise_shape, args,
                                            snr_low=args.train_enc_channel_low+2,
                                            snr_high=args.train_enc_channel_low+2)

            X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)


            dec1_self, dec1_other, dec2_self, dec2_other= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
            
            dec1_self = torch.clamp(dec1_self, 0.0, 1.0)
            dec1_other = torch.clamp(dec1_other, 0.0, 1.0)
            dec2_self = torch.clamp(dec2_self, 0.0, 1.0)
            dec2_other = torch.clamp(dec2_other, 0.0, 1.0)

            dec1_self = dec1_self.detach()
            dec1_other = dec1_other.detach()
            dec2_self = dec2_self.detach()
            dec2_other = dec2_other.detach()

            X_test_u1 = X_test_u1.detach()
            X_test_u2 = X_test_u2.detach()

            ##User 1
            #test_bce_loss_u1 += F.binary_cross_entropy(dec1_self, X_test_u1)
            #test_custom_loss_u1 += customized_loss(dec1_self, X_test_u1, noise = fwd_noise_u1, args = args)
            test_ber_u1  += errors_ber(dec1_self, X_test_u1)

            ##User 2
            #test_bce_loss_u2 += F.binary_cross_entropy(dec2_self, X_test_u2)
            #test_custom_loss_u2 += customized_loss(dec2_self, X_test_u2, noise = fwd_noise_u2, args = args)
            test_ber_u2  += errors_ber(dec2_self, X_test_u2)

    #test_bce_loss_u1 /= num_test_batch
    #test_custom_loss_u1 /= num_test_batch
    test_ber_u1  /= num_test_batch

    #test_bce_loss_u2 /= num_test_batch
    #test_custom_loss_u2 /= num_test_batch
    test_ber_u2  /= num_test_batch

    if verbose:
        print('====> User1 ber ', float(test_ber_u1),
        )

        print('====> User2 ber ', float(test_ber_u2),
        )

    # report_loss = float(test_bce_loss)
    # report_ber  = float(test_ber)

    # return report_loss, report_ber


def test_sic_MTL(model, args, block_len = 'default',use_cuda = False,ind=0, mockN=None):
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    
    block_len = args.block_len
    

    ber_res_u1, ber_res_u2, bler_res_u1,bler_res_u2 = [], [], [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    #snrs = [-1.5,0,1.5,3,6,9,12,15,18,21,24]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber_u1,test_ber_u2, test_bler_u1,test_bler_u2 = .0, .0, .0, .0
        with torch.no_grad():
            num_test_batch = int(5*args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test_u1     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                X_test_u2     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise_u1  = generate_noise(noise_shape, args, test_sigma=sigma)
                fwd_noise_u2  = generate_noise(noise_shape, args, test_sigma=sigma)

                X_test_u1, X_test_u2, fwd_noise_u1,fwd_noise_u2= X_test_u1.to(device),X_test_u2.to(device), fwd_noise_u1.to(device),fwd_noise_u2.to(device)

                
                dec1_self, dec1_other, dec2_self, dec2_other= model(X_test_u1,X_test_u2, fwd_noise_u1,fwd_noise_u2)
                

                ##User 1
                test_ber_u1  += errors_ber(dec1_self, X_test_u1)
                test_bler_u1 += errors_bler(dec1_self,X_test_u1)

                ##User 2
                test_ber_u2  += errors_ber(dec2_self, X_test_u2)
                test_bler_u2 += errors_bler(dec2_self,X_test_u2)


        test_ber_u1  /= num_test_batch
        test_bler_u1 /= num_test_batch

        test_ber_u2  /= num_test_batch
        test_bler_u2 /= num_test_batch       
        print('User1: Test SNR',this_snr ,'with ber ', float(test_ber_u1), 'with bler', float(test_bler_u1))
        print('User2: Test SNR',this_snr ,'with ber ', float(test_ber_u2), 'with bler', float(test_bler_u2))
        ber_res_u1.append(float(test_ber_u1))
        bler_res_u1.append( float(test_bler_u1))

        ber_res_u2.append(float(test_ber_u2))
        bler_res_u2.append( float(test_bler_u2))


    print('User 1 final results on SNRs ', snrs)
    print('BER', ber_res_u1)
    print('BLER', bler_res_u1)

    print('User 2 final results on SNRs ', snrs)
    print('BER', ber_res_u2)
    print('BLER', bler_res_u2)