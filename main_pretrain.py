##Train with some pretraining

import torch
import torch.optim as optim
import numpy as np
import sys
from get_args import get_args
from trainer_joint import train,train_with_trainable_intrlvr, validate, test
#from trainer_joint_custom_v2_v3_v4 import train, validate, test
from interleavers import Interleaver, DeInterleaver, Trainable_Interleaver

from numpy import arange
from numpy.random import mtrand
from channel_ae import Channel_Multiuser_Joint,Channel_Singleuser,Channel_Multiuser_Joint_v2,Channel_Multiuser_Joint_v3


def import_enc(args):

    from encoders import ENC_interCNN as ENC
    return ENC

def import_dec(args):

    from decoders import DEC_LargeCNN as DEC
    return DEC

if __name__ == '__main__':

    args = get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda"

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # choose encoder and decoder.
    ENC1 = import_enc(args)
    DEC1 = import_dec(args)

    ENC2 = import_enc(args)
    DEC2 = import_dec(args)

    delta=19    
    p_array_1=torch.zeros(args.block_len,dtype=torch.long)
    for i in range(args.block_len):
        p_array_1[i]=(i*delta)%(args.block_len)
    p_array_2 = p_array_1

    encoder1=ENC1(args, p_array_1).cuda()
    encoder2=ENC2(args, p_array_1).cuda()

    decoder1=DEC1(args, p_array_1).cuda()
    decoder2=DEC2(args, p_array_1).cuda()

    
    interleaver_u1          = Interleaver(args, p_array_1)
    interleaver_u2          = Interleaver(args, p_array_2)

    deinterleaver_u1        = DeInterleaver(args, p_array_1)
    deinterleaver_u2        = DeInterleaver(args, p_array_2)

    ##model to load
    pre_single=torch.load("./p2p_models_bl40/grid.pt")

    model_user1 = Channel_Singleuser(args, encoder1, decoder1, interleaver_u1, deinterleaver_u1).to(device)
    model_user2 = Channel_Singleuser(args, encoder2, decoder2, interleaver_u2, deinterleaver_u2).to(device)

    model_user1.load_state_dict(pre_single)
    model_user2.load_state_dict(pre_single)
    
    model = Channel_Multiuser_Joint(args, encoder1,encoder2, decoder1,decoder2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2).to(device)


    ##################################################################
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    OPT = optim.Adam
    encoder_params = list(encoder1.parameters()) + list(encoder2.parameters())
    decoder_params = list(decoder1.parameters()) + list(decoder2.parameters())



    encoders_optimizer=OPT(encoder_params ,lr=args.enc_lr)
    decoders_optimizer=OPT(filter(lambda p: p.requires_grad, decoder_params), lr=args.dec_lr)
        

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber = [], []
    alpha=0.5
    beta=0.5


    for epoch in range(1, args.num_epoch + 1):

        if args.num_train_enc > 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
            for idx in range(args.num_train_enc):

                if args.interleaver == "trainable":           
                    loss,loss1,loss2=train_with_trainable_intrlvr(alpha,beta,epoch, model, encoders_optimizer,intrlv_optimizer, args, use_cuda = use_cuda, mode ='encoder')
                else:
                    loss,loss1,loss2=train(alpha,beta,epoch, model, encoders_optimizer, args, use_cuda = use_cuda, mode ='encoder')
                    alpha = loss1/(loss1+loss2)
                    beta = loss2/(loss1+loss2)
                print("loss1=",loss1)
                print("loss2=",loss2)
                alpha=loss1/(loss1+loss2)
                beta=loss2/(loss2+loss1)   
                print("alpha=",alpha)
                print("beta=",beta)

        if args.num_train_dec > 0:
            for idx in range(args.num_train_dec):
                
                if args.interleaver == "trainable":           
                    loss,loss1,loss2=train_with_trainable_intrlvr(alpha,beta,epoch, model, decoders_optimizer,intrlv_optimizer, args, use_cuda = use_cuda, mode ='decoder')
                else:
                    loss,loss1,loss2=train(alpha,beta,epoch, model, decoders_optimizer, args, use_cuda = use_cuda, mode ='decoder')
                    alpha = loss1/(loss1+loss2)
                    beta = loss2/(loss1+loss2)
                
                print("loss1=",loss1)
                print("loss2=",loss2)       
                alpha=loss1/(loss1+loss2)
                beta=loss2/(loss2+loss1)

                print("alpha=",alpha)
                print("beta=",beta)
        

        if epoch%5 == 0:
            validate(model, args, use_cuda = use_cuda)
        


    #test the model
    test(model, args, use_cuda = use_cuda)








