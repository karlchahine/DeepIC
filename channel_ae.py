__author__ = 'yihanjiang'
import torch
import torch.nn.functional as F
from utils import snr_db2sigma
import numpy as np
import math
from numpy.linalg import inv

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

class Channel_Multiuser_Joint(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        if self.args.channel == 'awgn':
            signal1=codes1 + gamma1*codes2 + fwd_noise_u1
            signal2=codes2 + gamma2*codes1 + fwd_noise_u2

        elif self.args.channel == 'rayleigh':
            data_shape = codes1.shape
            fading_h_1 = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) 
            fading_h_1 = fading_h_1.type(torch.FloatTensor).to(self.this_device)
            fading_h_2 = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) 
            fading_h_2 = fading_h_2.type(torch.FloatTensor).to(self.this_device)

            signal1 = fading_h_1*codes1 + gamma1*fading_h_2*codes2 + fwd_noise_u1
            signal2 = fading_h_2*codes2 + gamma2*fading_h_1*codes1 + fwd_noise_u2

        elif self.args.channel == 'rician':
            data_shape = codes1.shape
            K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
            coeffLOS = np.sqrt(K/(K+1))
            coeffNLOS = np.sqrt(1/(K+1))
            hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
            hLOSImag = torch.ones(data_shape)

            ##u1
            hNLOSReal_1 = torch.randn(data_shape)
            hNLOSImag_1 = torch.randn(data_shape)
            fading_h_1 = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal_1,hNLOSImag_1)
            fading_h_1 = torch.abs(fading_h_1)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h_1 = fading_h_1.type(torch.FloatTensor).to(self.this_device)

            ##u2
            hNLOSReal_2 = torch.randn(data_shape)
            hNLOSImag_2 = torch.randn(data_shape)
            fading_h_2 = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal_2,hNLOSImag_2)
            fading_h_2 = torch.abs(fading_h_2)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h_2 = fading_h_2.type(torch.FloatTensor).to(self.this_device)

            signal1 = fading_h_1*codes1 + gamma1*fading_h_2*codes2 + fwd_noise_u1
            signal2 = fading_h_2*codes2 + gamma2*fading_h_1*codes1 + fwd_noise_u2

        elif self.args.channel == 'bursty':
            temp = np.random.binomial(1, .05, codes1.shape)
            temp2 = np.random.binomial(1, .05, codes1.shape)

            snr_bursty = 3
            sigma_bursty = snr_db2sigma(snr_bursty)

            noise_bursty = sigma_bursty*np.random.standard_normal(codes1.shape)
            noise_bursty=np.multiply(noise_bursty,temp)
            noise_bursty = torch.from_numpy(noise_bursty)
            noise_bursty=noise_bursty.type(torch.FloatTensor).to(self.this_device)

            noise_bursty2 = sigma_bursty*np.random.standard_normal(codes1.shape)
            noise_bursty2=np.multiply(noise_bursty2,temp2)
            noise_bursty2 = torch.from_numpy(noise_bursty2)
            noise_bursty2=noise_bursty2.type(torch.FloatTensor).to(self.this_device)

            signal1=codes1 + gamma1*codes2 + fwd_noise_u1 + noise_bursty
            signal2=codes2 + gamma2*codes1 + fwd_noise_u2 + noise_bursty2

        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2

class Channel_Multiuser_Joint_with_0(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint_with_0, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        device = torch.device("cuda")

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)
        codes2 = torch.zeros((codes1.shape)).to(device)
        #c = torch.zeros((codes1.shape[0],codes1.shape[1])).to(device)
        #codes2[:,:,1] = codes2[:,:,1]*c

        if self.args.channel == 'awgn':
            signal1=codes1 + gamma1*codes2 + fwd_noise_u1
            signal2=codes2 + gamma2*codes1 + fwd_noise_u2


        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2


class Channel_Multiuser_Joint_fix_enc(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint_fix_enc, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.mod1_u1 = torch.nn.Linear(args.block_len, args.block_len)
        self.mod2_u1 = torch.nn.Linear(args.block_len, args.block_len)
        self.mod3_u1 = torch.nn.Linear(args.block_len, args.block_len)

        self.mod1_u2 = torch.nn.Linear(args.block_len, args.block_len)
        self.mod2_u2 = torch.nn.Linear(args.block_len, args.block_len)
        self.mod3_u2 = torch.nn.Linear(args.block_len, args.block_len)

        self.mod1_u1.weight.data.copy_(torch.eye(args.block_len))
        self.mod2_u1.weight.data.copy_(torch.eye(args.block_len))
        self.mod3_u1.weight.data.copy_(torch.eye(args.block_len))
        self.mod1_u2.weight.data.copy_(torch.eye(args.block_len))
        self.mod2_u2.weight.data.copy_(torch.eye(args.block_len))
        self.mod3_u2.weight.data.copy_(torch.eye(args.block_len))

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        codes1 = torch.cat((self.mod1_u1(codes1[:,:,0]).unsqueeze(2), self.mod2_u1(codes1[:,:,1]).unsqueeze(2), self.mod3_u1(codes1[:,:,2]).unsqueeze(2)),2)
        codes2 = torch.cat((self.mod1_u2(codes2[:,:,0]).unsqueeze(2), self.mod2_u2(codes2[:,:,1]).unsqueeze(2), self.mod3_u2(codes2[:,:,2]).unsqueeze(2)),2)

        codes1 = F.elu(codes1)
        codes2 = F.elu(codes2)

        codes1 = (codes1-torch.mean(codes1))*1.0 / torch.std(codes1)
        codes2 = (codes2-torch.mean(codes2))*1.0 / torch.std(codes2)

        signal1=codes1 + gamma1*codes2 + fwd_noise_u1
        signal2=codes2 + gamma2*codes1 + fwd_noise_u2

        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2

class Channel_Multiuser_Joint_sic_MTL(torch.nn.Module):
    def __init__(self, args, enc1,enc2, shared1, shared2,interleaver,deinterleaver):
        super(Channel_Multiuser_Joint_sic_MTL, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.shared1 = shared1
        self.shared2 = shared2

        self.decoder11 = torch.nn.Linear(args.block_len, args.block_len)
        self.decoder12 = torch.nn.Linear(args.block_len, args.block_len)

        self.decoder21 = torch.nn.Linear(args.block_len, args.block_len)
        self.decoder22 = torch.nn.Linear(args.block_len, args.block_len)

        self.interleaver = interleaver
        self.deinterleaver = deinterleaver

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver)
        codes2 = self.encoder2(input_u2,self.interleaver)

        signal1=codes1 + gamma1*codes2 + fwd_noise_u1
        signal2=codes2 + gamma2*codes1 + fwd_noise_u2

        decoded1 = self.shared1(signal1, self.interleaver, self.deinterleaver)
        decoded2 = self.shared2(signal2, self.interleaver, self.deinterleaver)

        decoded1 = torch.reshape(decoded1, (self.args.batch_size, self.args.block_len))
        decoded2 = torch.reshape(decoded2, (self.args.batch_size, self.args.block_len))

        dec1_self = torch.sigmoid(self.decoder11(decoded1))
        dec1_other = torch.sigmoid(self.decoder12(decoded1))

        dec2_self = torch.sigmoid(self.decoder21(decoded2))
        dec2_other = torch.sigmoid(self.decoder22(decoded2))

        dec1_self = torch.reshape(dec1_self, (self.args.batch_size, self.args.block_len, 1))
        dec1_other = torch.reshape(dec1_other, (self.args.batch_size, self.args.block_len, 1))
        dec2_self = torch.reshape(dec2_self, (self.args.batch_size, self.args.block_len, 1))
        dec2_other = torch.reshape(dec2_other, (self.args.batch_size, self.args.block_len, 1))

        return dec1_self, dec1_other, dec2_self, dec2_other

        #return decoded1, decoded1, decoded2, decoded2

class Channel_Multiuser_Joint_sic(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,reenc1,reenc2,finaldec1,finaldec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint_sic, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.reencoder1 = reenc1
        self.reencoder2 = reenc2

        self.finaldecoder1 = finaldec1
        self.finaldecoder2 = finaldec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

       
        signal1=codes1 + gamma1*codes2 + fwd_noise_u1
        signal2=codes2 + gamma2*codes1 + fwd_noise_u2

        ##This is the message of the other user
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)

        reencoded1 = self.reencoder1(decoded1,self.interleaver_u1)
        reencoded2 = self.reencoder2(decoded2,self.interleaver_u2)

        final_decoded1 = self.finaldecoder1(signal1 - gamma1*reencoded1,self.interleaver_u1,self.deinterleaver_u1)
        final_decoded2 = self.finaldecoder2(signal2 - gamma2*reencoded2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2, final_decoded1, final_decoded2

#don't use h in estimation
class Channel_Multiuser_Joint_sic_v2(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,reenc1,reenc2,finaldec1,finaldec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint_sic_v2, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.reencoder1 = reenc1
        self.reencoder2 = reenc2

        self.finaldecoder1 = finaldec1
        self.finaldecoder2 = finaldec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.comb1 = torch.nn.Linear(2*args.block_len, args.block_len)
        self.comb2 = torch.nn.Linear(2*args.block_len, args.block_len)
        self.comb3 = torch.nn.Linear(2*args.block_len, args.block_len)

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        signal1=codes1 + gamma1*codes2 + fwd_noise_u1
        signal2=codes2 + gamma2*codes1 + fwd_noise_u2

        ##This is the message of the other user
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)

        reencoded1 = self.reencoder1(decoded1,self.interleaver_u1)
        reencoded2 = self.reencoder2(decoded2,self.interleaver_u2)

        temp1_1 = torch.unsqueeze(self.comb1(torch.cat((signal1[:,:,0], gamma1*reencoded1[:,:,0]),1)),2)
        temp1_2 = torch.unsqueeze(self.comb2(torch.cat((signal1[:,:,1], gamma1*reencoded1[:,:,1]),1)),2)
        temp1_3 = torch.unsqueeze(self.comb3(torch.cat((signal1[:,:,2], gamma1*reencoded1[:,:,2]),1)),2)

        temp1 = torch.cat((temp1_1,temp1_2,temp1_3),2)

        temp2_1 = torch.unsqueeze(self.comb1(torch.cat((signal2[:,:,0], gamma2*reencoded2[:,:,0]),1)),2)
        temp2_2 = torch.unsqueeze(self.comb2(torch.cat((signal2[:,:,1], gamma2*reencoded2[:,:,1]),1)),2)
        temp2_3 = torch.unsqueeze(self.comb3(torch.cat((signal2[:,:,2], gamma2*reencoded2[:,:,2]),1)),2)

        temp2 = torch.cat((temp2_1,temp2_2,temp2_3),2)

        final_decoded1 = self.finaldecoder1(temp1,self.interleaver_u1,self.deinterleaver_u1)
        final_decoded2 = self.finaldecoder2(temp2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2, final_decoded1, final_decoded2


##fix encoders as turbo
class Channel_Multiuser_Joint_v2(torch.nn.Module):
    def __init__(self, args, dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint_v2, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2,puncture=False,pseudo=False):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = input_u1
        codes2 = input_u2



        # codes1 = torch.transpose(input_u1,1,2)
        # codes1 = torch.flatten(codes1,start_dim = 1)

        # codes2 = torch.transpose(input_u2,1,2)
        # codes2 = torch.flatten(codes2,start_dim = 1)

        # codes1 = self.lin1(codes1)
        # codes2 = self.lin1(codes2)

        # codes1 = torch.reshape(codes1,(self.args.batch_size,self.args.block_len,3))
        # codes2 = torch.reshape(codes2,(self.args.batch_size,self.args.block_len,3))

        # codes1 = self.power_constraint(codes1)
        # codes2 = self.power_constraint(codes2)


        

        if self.args.channel == 'awgn':
            signal1=codes1 + gamma1*codes2 + fwd_noise_u1
            signal2=codes2 + gamma2*codes1 + fwd_noise_u2
            

        elif self.args.channel == 'rayleigh':
            data_shape = codes1.shape
            fading_h_1 = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) 
            fading_h_1 = fading_h_1.type(torch.FloatTensor).to(self.this_device)
            fading_h_2 = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) 
            fading_h_2 = fading_h_2.type(torch.FloatTensor).to(self.this_device)

            signal1 = fading_h_1*codes1 + gamma1*fading_h_2*codes2 + fwd_noise_u1
            signal2 = fading_h_2*codes2 + gamma2*fading_h_1*codes1 + fwd_noise_u2

        elif self.args.channel == 'rician':
            data_shape = codes1.shape
            K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
            coeffLOS = np.sqrt(K/(K+1))
            coeffNLOS = np.sqrt(1/(K+1))
            hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
            hLOSImag = torch.ones(data_shape)

            ##u1
            hNLOSReal_1 = torch.randn(data_shape)
            hNLOSImag_1 = torch.randn(data_shape)
            fading_h_1 = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal_1,hNLOSImag_1)
            fading_h_1 = torch.abs(fading_h_1)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h_1 = fading_h_1.type(torch.FloatTensor).to(self.this_device)

            ##u2
            hNLOSReal_2 = torch.randn(data_shape)
            hNLOSImag_2 = torch.randn(data_shape)
            fading_h_2 = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal_2,hNLOSImag_2)
            fading_h_2 = torch.abs(fading_h_2)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h_2 = fading_h_2.type(torch.FloatTensor).to(self.this_device)

            signal1 = fading_h_1*codes1 + gamma1*fading_h_2*codes2 + fwd_noise_u1
            signal2 = fading_h_2*codes2 + gamma2*fading_h_1*codes1 + fwd_noise_u2

        elif self.args.channel == 'bursty':
            temp = np.random.binomial(1, .05, codes1.shape)
            temp2 = np.random.binomial(1, .05, codes1.shape)

            snr_bursty = 3
            sigma_bursty = snr_db2sigma(snr_bursty)

            noise_bursty = sigma_bursty*np.random.standard_normal(codes1.shape)
            noise_bursty=np.multiply(noise_bursty,temp)
            noise_bursty = torch.from_numpy(noise_bursty)
            noise_bursty=noise_bursty.type(torch.FloatTensor).to(self.this_device)

            noise_bursty2 = sigma_bursty*np.random.standard_normal(codes1.shape)
            noise_bursty2=np.multiply(noise_bursty2,temp2)
            noise_bursty2 = torch.from_numpy(noise_bursty2)
            noise_bursty2=noise_bursty2.type(torch.FloatTensor).to(self.this_device)

            signal1=codes1 + gamma1*codes2 + fwd_noise_u1 + noise_bursty
            signal2=codes2 + gamma2*codes1 + fwd_noise_u2 + noise_bursty2
            

        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2


##fix encoders as TurboAE, train mod/demod/decoder
class Channel_Multiuser_Joint_v3(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Channel_Multiuser_Joint_v3, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        # self.mod1 = torch.nn.Linear(3*args.block_len, 3*args.block_len)
        # self.mod2 = torch.nn.Linear(3*args.block_len, 3*args.block_len)

        # self.demod1 = torch.nn.Linear(3*args.block_len, 3*args.block_len)
        # self.demod2 = torch.nn.Linear(3*args.block_len, 3*args.block_len)

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std
        return x_input_norm
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        # codeword1=torch.transpose(codes1, 1, 2)
        # codeword1=torch.reshape(codeword1, (self.args.batch_size,3*self.args.block_len))

        # codeword2=torch.transpose(codes2, 1, 2)
        # codeword2=torch.reshape(codeword2, (self.args.batch_size,3*self.args.block_len))

        # codes1 = F.elu(self.mod1(codeword1))
        # codes2 = F.elu(self.mod2(codeword2))

        # codes1 = torch.reshape(codes1, (self.args.batch_size, 3, self.args.block_len))
        # codes2 = torch.reshape(codes2, (self.args.batch_size, 3, self.args.block_len))
        # codes1=torch.transpose(codes1, 1, 2)
        # codes2=torch.transpose(codes2, 1, 2)

        #codes1 = self.power_constraint(codes1)
        #codes2 = self.power_constraint(codes2)

        signal1=codes1 + gamma1*codes2 + fwd_noise_u1
        signal2=codes2 + gamma2*codes1 + fwd_noise_u2

        # signal1=torch.transpose(signal1, 1, 2)
        # signal1=torch.reshape(signal1, (self.args.batch_size,3*self.args.block_len))

        # signal2=torch.transpose(signal2, 1, 2)
        # signal2=torch.reshape(signal2, (self.args.batch_size,3*self.args.block_len))

        # signal1 = self.demod1(signal1)
        # signal2 = self.demod2(signal2)

        # signal1 = torch.reshape(signal1, (self.args.batch_size, 3, self.args.block_len))
        # signal2 = torch.reshape(signal2, (self.args.batch_size, 3, self.args.block_len))
        # signal1=torch.transpose(signal1, 1, 2)
        # signal2=torch.transpose(signal2, 1, 2)

        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2


##Complex channels for review
class Channel_Multiuser_Joint_v4(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2,mod1,mod2,demod1,demod2):
        super(Channel_Multiuser_Joint_v4, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.modulator1 = mod1
        self.modulator2 = mod2

        self.demodulator1 = demod1
        self.demodulator2 = demod2

        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std
        return x_input_norm
    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        device = "cuda"

        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        
        codes1=torch.transpose(codes1, 1, 2)
        codes1=torch.reshape(codes1, (self.args.batch_size,3*self.args.block_len))
        codes1=torch.reshape(codes1, (self.args.batch_size,3*self.args.block_len//2, 2))

        mod_sig1 = self.modulator1(codes1)

        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        codes2=torch.transpose(codes2, 1, 2)
        codes2=torch.reshape(codes2, (self.args.batch_size,3*self.args.block_len))
        codes2=torch.reshape(codes2, (self.args.batch_size,3*self.args.block_len//2, 2))

        mod_sig2 = self.modulator2(codes2)


        fwd_noise_u1 =  torch.reshape(fwd_noise_u1, (self.args.batch_size,3*self.args.block_len//2, 2))
        fwd_noise_u2 =  torch.reshape(fwd_noise_u2, (self.args.batch_size,3*self.args.block_len//2, 2))

        x1r = mod_sig1[:,:,0]
        x1i = mod_sig1[:,:,1]

        x2r = mod_sig2[:,:,0]
        x2i = mod_sig2[:,:,1]

        y1r = 1*x1r - 0*x1i + 0.4*x2r - 0.7*x2i + fwd_noise_u1[:,:,0]
        y1i = 0*x1r + 1*x1i + 0.7*x2r + 0.4*x2i + fwd_noise_u1[:,:,1]
        y1r = torch.unsqueeze(y1r, 2)
        y1i = torch.unsqueeze(y1i, 2)
        signal1 = torch.cat((y1r, y1i), dim = 2)

        y2r = 1*x2r - 0*x2i + 0.4*x1r - 0.7*x1i + fwd_noise_u2[:,:,0]
        y2i = 0*x2r + 1*x2i + 0.7*x1r + 0.4*x1i + fwd_noise_u2[:,:,1]
        y2r = torch.unsqueeze(y2r, 2)
        y2i = torch.unsqueeze(y2i, 2)
        signal2 = torch.cat((y2r, y2i), dim = 2)

        demod_sig1 = self.demodulator1(signal1)
        demod_sig1 = torch.reshape(demod_sig1,(self.args.batch_size,3*self.args.block_len))
        demod_sig1 = torch.reshape(demod_sig1,(self.args.batch_size,3,self.args.block_len))
        demod_sig1=torch.transpose(demod_sig1, 1, 2)

        demod_sig2 = self.demodulator2(signal2)
        demod_sig2 = torch.reshape(demod_sig2,(self.args.batch_size,3*self.args.block_len))
        demod_sig2 = torch.reshape(demod_sig2,(self.args.batch_size,3,self.args.block_len))
        demod_sig2=torch.transpose(demod_sig2, 1, 2)


        decoded1=self.decoder1(demod_sig1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(demod_sig2,self.interleaver_u2,self.deinterleaver_u2)
        

        return decoded1, decoded2


##Used to finetune DeepIC+, to be deleted later
class Channel_Singleuser_test(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_Singleuser_test, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise, sigma=0):
        
        codes  = self.enc(input)
        received_codes = codes + fwd_noise       
        x_dec          = self.dec(received_codes)

        return x_dec, codes


class Channel_Singleuser(torch.nn.Module):
    def __init__(self, args, enc,dec,interleaver,deinterleaver):
        super(Channel_Singleuser, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder = enc

        self.decoder = dec

        self.interleaver = interleaver
        self.deinterleaver = deinterleaver

    def forward(self, input, fwd_noise):        
        
        codes = self.encoder(input,self.interleaver)

        if self.args.channel =='awgn':
            signal = codes + fwd_noise

        elif self.args.channel == 'rician':

            data_shape = codes.shape
            K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
            coeffLOS = np.sqrt(K/(K+1))
            coeffNLOS = np.sqrt(1/(K+1))

            hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
            hLOSImag = torch.ones(data_shape)
            hNLOSReal = torch.randn(data_shape)
            hNLOSImag = torch.randn(data_shape)      

            fading_h = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal,hNLOSImag)
            fading_h = torch.abs(fading_h)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)

            signal = fading_h*codes + fwd_noise 
        
        decoded=self.decoder(signal,self.interleaver,self.deinterleaver)
        
        return decoded

class Channel_Singleuser_with_mod(torch.nn.Module):
    def __init__(self, args, enc,dec,interleaver,deinterleaver, mod, demod):
        super(Channel_Singleuser_with_mod, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder = enc

        self.decoder = dec

        self.modulator = mod
        self.demodulator = demod

        self.interleaver = interleaver
        self.deinterleaver = deinterleaver

    def forward(self, input, fwd_noise):        
        
        codes = self.encoder(input,self.interleaver)

        codes=torch.transpose(codes, 1, 2)
        codes=torch.reshape(codes, (self.args.batch_size,3*self.args.block_len))
        codes=torch.reshape(codes, (self.args.batch_size,3*self.args.block_len//2, 2))

        mod_sig = self.modulator(codes)
        fwd_noise=torch.reshape(fwd_noise, (self.args.batch_size,3*self.args.block_len//2, 2))

        signal = mod_sig + fwd_noise
        demod_sig = self.demodulator(signal)
        demod_sig = torch.reshape(demod_sig,(self.args.batch_size,3*self.args.block_len))

        demod_sig = torch.reshape(demod_sig,(self.args.batch_size,3,self.args.block_len))
        demod_sig=torch.transpose(demod_sig, 1, 2)
        
        decoded=self.decoder(demod_sig,self.interleaver,self.deinterleaver)
        
        return decoded



class Channel_3users(torch.nn.Module):
    def __init__(self, args, enc1,enc2,enc3,dec1,dec2,dec3,interleaver_u1,interleaver_u2,interleaver_u3,deinterleaver_u1,deinterleaver_u2,deinterleaver_u3):
        super(Channel_3users, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2
        self.encoder3 = enc3

        self.decoder1 = dec1
        self.decoder2 = dec2
        self.decoder3 = dec3

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.interleaver_u3 = interleaver_u3

        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2
        self.deinterleaver_u3 = deinterleaver_u3

        self.gamma = args.gamma

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2,input_u3, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3):

        
        gamma = self.gamma
        
        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)
        codes3 = self.encoder3(input_u3,self.interleaver_u3)

        signal1=codes1 + gamma*(codes2+codes3) + fwd_noise_u1
        signal2=codes2 + gamma*(codes1+codes3) + fwd_noise_u2
        signal3=codes3 + gamma*(codes1+codes2) + fwd_noise_u3
            

        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        decoded3=self.decoder3(signal3,self.interleaver_u3,self.deinterleaver_u3)
        

        return decoded1, decoded2, decoded3


class Channel_4users(torch.nn.Module): ##batch size = 250
    def __init__(self, args, enc1,enc2,enc3,enc4,dec1,dec2,dec3,dec4,interleaver_u1,interleaver_u2,interleaver_u3,interleaver_u4,deinterleaver_u1,deinterleaver_u2,deinterleaver_u3,deinterleaver_u4):
        super(Channel_4users, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2
        self.encoder3 = enc3
        self.encoder4 = enc4

        self.decoder1 = dec1
        self.decoder2 = dec2
        self.decoder3 = dec3
        self.decoder4 = dec4

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.interleaver_u3 = interleaver_u3
        self.interleaver_u4 = interleaver_u4

        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2
        self.deinterleaver_u3 = deinterleaver_u3
        self.deinterleaver_u4 = deinterleaver_u4

        self.gamma = args.gamma

    def set_weights(self, x,y):
        self.alpha=x
        self.beta=y
    
    def forward(self, input_u1,input_u2,input_u3,input_u4, fwd_noise_u1,fwd_noise_u2,fwd_noise_u3,fwd_noise_u4):

        
        gamma = self.gamma
        
        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)
        codes3 = self.encoder3(input_u3,self.interleaver_u3)
        codes4 = self.encoder4(input_u4,self.interleaver_u4)

        signal1=codes1 + gamma*(codes2+codes3+codes4) + fwd_noise_u1
        signal2=codes2 + gamma*(codes1+codes3+codes4) + fwd_noise_u2
        signal3=codes3 + gamma*(codes1+codes2+codes4) + fwd_noise_u3
        signal4=codes4 + gamma*(codes1+codes2+codes3) + fwd_noise_u4
            

        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        decoded3=self.decoder3(signal3,self.interleaver_u3,self.deinterleaver_u3)
        decoded4=self.decoder4(signal4,self.interleaver_u4,self.deinterleaver_u4)
        

        return decoded1, decoded2, decoded3, decoded4

























        

class Channel_AE(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_AE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise):
  
        codes  = self.enc(input)

        # Setup channel mode:
        if self.args.channel in ['awgn', 't-dist', 'radar', 'ge_awgn']:
            received_codes = codes + fwd_noise

        
        elif self.args.channel in ['rician', 'rician_coherent']:
            data_shape = codes.shape
            K = self.args.rician_K
            coeffLOS = np.sqrt(K/(K+1))
            coeffNLOS = np.sqrt(1/(K+1))

            hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
            hLOSImag = torch.ones(data_shape)
            hNLOSReal = torch.randn(data_shape)
            hNLOSImag = torch.randn(data_shape)
          

            fading_h = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal,hNLOSImag)
            #Assuming phase information at the receiver
            fading_h = torch.abs(fading_h)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)

            received_codes = fading_h*codes + fwd_noise
            



        else:
            print('default AWGN channel')
            received_codes = codes + fwd_noise

  

        x_dec          = self.dec(received_codes)

        return x_dec, codes



##Broadcast stuff

##Classical broadcast
class Broadcast_Channel(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(2*3*args.block_len,3*args.block_len)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        
        
        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        codes1 = torch.transpose(codes1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(codes2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        #codes1 = torch.reshape(codes1,(codes1.shape[0],codes1.shape[1]*codes1.shape[2]))
        #codes2 = torch.reshape(codes2,(codes2.shape[0],codes2.shape[1]*codes2.shape[2]))



        concatenated_code = torch.cat((codes1,codes2),1)

        combined_signal = self.power_constraint(self.combiner(concatenated_code))
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: fix turbo encoding, combiner is NN 
class Broadcast_Channel_v2(torch.nn.Module):
    def __init__(self, args,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v2, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(2*3*args.block_len,3*args.block_len)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        if self.args.train_channel_mode == 'block_norm_ste':
            stequantize = STEQuantize.apply
            x_input_norm = stequantize(x_input_norm, self.args)

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2,corrupt_comb=False,start=0,puncture=False,pseudo=False):

        device = torch.device("cuda")


    

        

        codes1 = torch.transpose(input_u1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(input_u2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)




        if puncture:
            temp = torch.zeros((codes1.shape[0],39),dtype=torch.float).to(device)
         
            codes1[:,41:80] = torch.mul(codes1[:,41:80],temp).to(device)
            codes1[:,81:120] = torch.mul(codes1[:,81:120],temp).to(device)


            codes2[:,41:80] = torch.mul(codes2[:,41:80],temp).to(device)
            codes2[:,81:120] = torch.mul(codes2[:,81:120],temp).to(device)

           
        concatenated_code = torch.cat((codes1,codes2),1)
        concatenated_code = self.combiner(concatenated_code)

        

        if corrupt_comb:
            stride = 10
            noise = 100*torch.randn((concatenated_code.shape[0],stride), dtype=torch.float).to(device)
            concatenated_code[:,start:start + stride] = concatenated_code[:,start:start + stride] + noise

        combined_signal = self.power_constraint(concatenated_code)
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2

        if pseudo:

            W = self.combiner.weight.cpu().detach().numpy()

            W_effective_1 = W[:,0:41]
            W_effective_2 = W[:,80]
            W_effective_3 = W[:,120:161]
            W_effective_4 = W[:,200]

            # W_effective = np.concatenate((W_effective_1,W_effective_2,W_effective_3,W_effective_4),1)
            W_effective = np.concatenate((W_effective_1,W_effective_2,W_effective_3,W_effective_4),1)

            # a = W_effective_1.transpose()
            # a = np.matmul(a,W_effective_1)
            # a = inv(a)
            # W_effective_1_pseudo = np.matmul(a,W_effective_1.transpose())

            # b = W_effective_2.transpose()
            # b = np.matmul(b,W_effective_2)
            # b = inv(b)
            # W_effective_2_pseudo = np.matmul(b,W_effective_2.transpose())

            a = W_effective.transpose()
            a = np.matmul(a,W_effective)
            a = inv(a)
            W_effective_pseudo = np.matmul(a,W_effective.transpose())

            W_effective_pseudo = np.linalg.pinv(W_effective)

            signal1 = torch.transpose(signal1,1,2)
            signal1 = torch.flatten(signal1,start_dim = 1)

            signal2 = torch.transpose(signal2,1,2)
            signal2 = torch.flatten(signal2,start_dim = 1)


            # decoded1 = np.matmul(signal1.cpu().detach().numpy(),W_effective_1_pseudo.transpose())
            # decoded2 = np.matmul(signal2.cpu().detach().numpy(),W_effective_2_pseudo.transpose())

            decoded1 = np.matmul(signal1.cpu().detach().numpy(),W_effective_pseudo.transpose())
            decoded2 = np.matmul(signal2.cpu().detach().numpy(),W_effective_pseudo.transpose())

            decoded1 = decoded1[:,0:40]
            decoded2 = decoded2[:,40:80]


            ##get 80, then first 40 are for user 1 last 40 are for user 2

            decoded1 = np.reshape(decoded1,(500,40,1))
            decoded2 = np.reshape(decoded2,(500,40,1))

            decoded1 = torch.from_numpy(decoded1).to(device)
            decoded2 = torch.from_numpy(decoded2).to(device)

            return decoded1, decoded2

        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2



##Broadcast: fix turbo encoding, combiner consists of 2 weights 
class Broadcast_Channel_v3(torch.nn.Module):
    def __init__(self, args,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v3, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        # self.W_1 = torch.nn.parameter.Parameter(torch.randn(1,device="cuda"))
        # self.W_2 = torch.nn.parameter.Parameter(torch.randn(1,device="cuda"))

        self.W_1 = torch.randn((1), device = "cuda",requires_grad=True)
        self.W_2 = torch.randn((1), device = "cuda",requires_grad=True)


    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        

        codes1 = torch.transpose(input_u1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(input_u2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        combined_signal = self.power_constraint((self.W_1)*codes1 + (self.W_2)*codes2)
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: fix turbo encoding + combining, just learn decoder 
class Broadcast_Channel_v4(torch.nn.Module):
    def __init__(self, args,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v4, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2


    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        

        codes1 = torch.transpose(input_u1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(input_u2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        combined_signal = self.power_constraint(np.sqrt(0.7)*codes1 + np.sqrt(0.3)*codes2)
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: fix turbo encoding, combiner is NN, remove parities
class Broadcast_Channel_v5(torch.nn.Module):
    def __init__(self, args,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v5, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(80,120)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        if self.args.train_channel_mode == 'block_norm_ste':
            stequantize = STEQuantize.apply
            x_input_norm = stequantize(x_input_norm, self.args)

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2,corrupt_comb=False,start=0,puncture=False,pseudo=False):

        device = torch.device("cuda")

        codes1 = torch.transpose(input_u1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(input_u2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        codes1 = codes1[:,:40]
        codes2 = codes2[:,:40]

        if puncture:
            temp = torch.zeros((codes1.shape[0],39),dtype=torch.float).to(device)
         
            codes1[:,41:80] = torch.mul(codes1[:,41:80],temp).to(device)
            codes1[:,81:120] = torch.mul(codes1[:,81:120],temp).to(device)


            codes2[:,41:80] = torch.mul(codes2[:,41:80],temp).to(device)
            codes2[:,81:120] = torch.mul(codes2[:,81:120],temp).to(device)

           
        concatenated_code = torch.cat((codes1,codes2),1)
        concatenated_code = self.combiner(concatenated_code)

        

        if corrupt_comb:
            stride = 10
            noise = 100*torch.randn((concatenated_code.shape[0],stride), dtype=torch.float).to(device)
            concatenated_code[:,start:start + stride] = concatenated_code[:,start:start + stride] + noise

        combined_signal = self.power_constraint(concatenated_code)
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2

        if pseudo:

            W = self.combiner.weight.cpu().detach().numpy()

            W_effective_1 = W[:,0:41]
            W_effective_2 = W[:,80]
            W_effective_3 = W[:,120:161]
            W_effective_4 = W[:,200]

            # W_effective = np.concatenate((W_effective_1,W_effective_2,W_effective_3,W_effective_4),1)
            W_effective = np.concatenate((W_effective_1,W_effective_2,W_effective_3,W_effective_4),1)

            # a = W_effective_1.transpose()
            # a = np.matmul(a,W_effective_1)
            # a = inv(a)
            # W_effective_1_pseudo = np.matmul(a,W_effective_1.transpose())

            # b = W_effective_2.transpose()
            # b = np.matmul(b,W_effective_2)
            # b = inv(b)
            # W_effective_2_pseudo = np.matmul(b,W_effective_2.transpose())

            a = W_effective.transpose()
            a = np.matmul(a,W_effective)
            a = inv(a)
            W_effective_pseudo = np.matmul(a,W_effective.transpose())

            W_effective_pseudo = np.linalg.pinv(W_effective)

            signal1 = torch.transpose(signal1,1,2)
            signal1 = torch.flatten(signal1,start_dim = 1)

            signal2 = torch.transpose(signal2,1,2)
            signal2 = torch.flatten(signal2,start_dim = 1)


            # decoded1 = np.matmul(signal1.cpu().detach().numpy(),W_effective_1_pseudo.transpose())
            # decoded2 = np.matmul(signal2.cpu().detach().numpy(),W_effective_2_pseudo.transpose())

            decoded1 = np.matmul(signal1.cpu().detach().numpy(),W_effective_pseudo.transpose())
            decoded2 = np.matmul(signal2.cpu().detach().numpy(),W_effective_pseudo.transpose())

            decoded1 = decoded1[:,0:40]
            decoded2 = decoded2[:,40:80]


            ##get 80, then first 40 are for user 1 last 40 are for user 2

            decoded1 = np.reshape(decoded1,(500,40,1))
            decoded2 = np.reshape(decoded2,(500,40,1))

            decoded1 = torch.from_numpy(decoded1).to(device)
            decoded2 = torch.from_numpy(decoded2).to(device)

            return decoded1, decoded2

        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Neural encoding, combining and decoding. But received signals have different statistics
class Broadcast_Channel_v6(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v6, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(2*3*args.block_len,3*args.block_len)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        
        
        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        codes1 = torch.transpose(codes1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(codes2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        #codes1 = torch.reshape(codes1,(codes1.shape[0],codes1.shape[1]*codes1.shape[2]))
        #codes2 = torch.reshape(codes2,(codes2.shape[0],codes2.shape[1]*codes2.shape[2]))



        concatenated_code = torch.cat((codes1,codes2),1)

        combined_signal = self.power_constraint(self.combiner(concatenated_code))
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = 3*combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: fix turbo encoding, combiner is NN, but received signals have different statistics
class Broadcast_Channel_v7(torch.nn.Module):
    def __init__(self, args,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v7, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(2*3*args.block_len,3*args.block_len)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        if self.args.train_channel_mode == 'block_norm_ste':
            stequantize = STEQuantize.apply
            x_input_norm = stequantize(x_input_norm, self.args)

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2,puncture=False):

        device = torch.device("cuda")
        codes1 = torch.transpose(input_u1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(input_u2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        concatenated_code = torch.cat((codes1,codes2),1)
        concatenated_code = self.combiner(concatenated_code)


        combined_signal = self.power_constraint(concatenated_code)
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = 3*combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Neural encoding, combining and decoding. But received signals have different statistics
class Broadcast_Channel_v6(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v6, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(2*3*args.block_len,3*args.block_len)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        
        
        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        codes1 = torch.transpose(codes1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(codes2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        #codes1 = torch.reshape(codes1,(codes1.shape[0],codes1.shape[1]*codes1.shape[2]))
        #codes2 = torch.reshape(codes2,(codes2.shape[0],codes2.shape[1]*codes2.shape[2]))



        concatenated_code = torch.cat((codes1,codes2),1)

        combined_signal = self.power_constraint(self.combiner(concatenated_code))
        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = 3*combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: everything is neural, combiner is CNN
class Broadcast_Channel_v8(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v8, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

        self.combiner = torch.nn.Linear(2*3*args.block_len,3*args.block_len)
        self.combiner_2 = torch.nn.Linear(3*args.block_len,3*args.block_len)

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        
        
        codes1 = self.encoder1(input_u1,self.interleaver_u1)
        codes2 = self.encoder2(input_u2,self.interleaver_u2)

        codes1 = torch.transpose(codes1,1,2)
        codes1 = torch.flatten(codes1,start_dim = 1)

        codes2 = torch.transpose(codes2,1,2)
        codes2 = torch.flatten(codes2,start_dim = 1)

        #codes1 = torch.reshape(codes1,(codes1.shape[0],codes1.shape[1]*codes1.shape[2]))
        #codes2 = torch.reshape(codes2,(codes2.shape[0],codes2.shape[1]*codes2.shape[2]))



        concatenated_code = torch.cat((codes1,codes2),1)
        concatenated_code = F.elu(self.combiner(concatenated_code))
        concatenated_code = F.elu(self.combiner_2(concatenated_code))
        combined_signal = self.power_constraint(concatenated_code)

        combined_signal = torch.reshape(combined_signal,(self.args.batch_size,self.args.block_len,3))

        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: 2 encoders, 2 decoders
class Broadcast_Channel_v9(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v9, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        if self.args.train_channel_mode == 'block_norm_ste':
            stequantize = STEQuantize.apply
            x_input_norm = stequantize(x_input_norm, self.args)

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        code1 = self.encoder1(input_u1, self.interleaver_u1)/(np.sqrt(2))
        code2 = self.encoder2(input_u2, self.interleaver_u2)/(np.sqrt(2))

        combined_signal = code1 + code2
   
        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: 2 encoders, 2 decoders, 2 weights
class Broadcast_Channel_v10(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v10, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args

        self.w1 = torch.nn.Parameter(torch.randn(()))
        self.w2 = torch.nn.Parameter(torch.randn(()))
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        denom = torch.sqrt(self.w1**2 + self.w2**2)

        code1 = self.encoder1(input_u1, self.interleaver_u1)
        code1 = self.w1*code1/denom

        code2 = self.encoder2(input_u2, self.interleaver_u2)
        code2 = self.w2*code2/denom

        combined_signal = code1 + code2
   
        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2


##Broadcast: 2 encoders, 2 decoders binarized
class Broadcast_Channel_v11(torch.nn.Module):
    def __init__(self, args, enc1,enc2,dec1,dec2,interleaver_u1,interleaver_u2,deinterleaver_u1,deinterleaver_u2):
        super(Broadcast_Channel_v11, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        
        self.encoder1 = enc1
        self.encoder2 = enc2

        self.decoder1 = dec1
        self.decoder2 = dec2

        self.interleaver_u1 = interleaver_u1
        self.interleaver_u2 = interleaver_u2
        self.deinterleaver_u1 = deinterleaver_u1
        self.deinterleaver_u2 = deinterleaver_u2

    def power_constraint(self, x_input):

        this_mean    = torch.mean(x_input)
        this_std     = torch.std(x_input)

        x_input_norm = (x_input-this_mean)*1.0 / this_std

        stequantize = STEQuantize.apply
        x_input_norm = stequantize(x_input_norm, self.args)

        return x_input_norm


    
    def forward(self, input_u1,input_u2, fwd_noise_u1,fwd_noise_u2):

        code1 = self.encoder1(input_u1, self.interleaver_u1)
        code2 = self.encoder2(input_u2, self.interleaver_u2)

        combined_signal = code1 + code2
        combined_signal = self.power_constraint(combined_signal)
   
        signal1 = combined_signal  + fwd_noise_u1
        signal2 = combined_signal + fwd_noise_u2
        
        decoded1=self.decoder1(signal1,self.interleaver_u1,self.deinterleaver_u1)
        decoded2=self.decoder2(signal2,self.interleaver_u2,self.deinterleaver_u2)
        
        return decoded1, decoded2