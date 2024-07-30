__author__ = 'hebbarashwin'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import csv
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

from turbo import turbo_encode, turbo_decode, bcjr_decode
from utils import snr_db2sigma, errors_ber, errors_bler, corrupt_signal, moving_average, Equalizer

class Turbo_subnet(nn.Module):
    def __init__(self, block_len, init_type = 'ones', non_linear = False, one_weight = False):
        super(Turbo_subnet, self).__init__()

        assert init_type in ['ones', 'random', 'gaussian'], "Invalid init type"
        self.non_linear = non_linear
        self.non_linearity = nn.LeakyReLU(0.1)
        if init_type == 'ones':
            self.w1 = nn.parameter.Parameter(torch.ones((1, block_len)))
            self.w2 = nn.parameter.Parameter(torch.ones((1, block_len)))
            self.w3 = nn.parameter.Parameter(torch.ones((1, block_len)))
        elif init_type == 'random':
            self.w1 = nn.parameter.Parameter(torch.rand((1, block_len)))
            self.w2 = nn.parameter.Parameter(torch.rand((1, block_len)))
            self.w3 = nn.parameter.Parameter(torch.rand((1, block_len)))
        elif init_type == 'gaussian':
            self.w1 = nn.parameter.Parameter(0.001* torch.randn((1, block_len)))
            self.w2 = nn.parameter.Parameter(0.001*torch.randn((1, block_len)))
            self.w3 = nn.parameter.Parameter(0.001*torch.randn((1, block_len)))

        if one_weight:
            self.w3 = self.w1
            self.w2 = self.w1

    def forward(self, L_ext, L_sys, L_int):

        if not hasattr(self, 'non_linear') or not self.non_linear:
            x = self.w1 * L_ext - self.w2 * L_sys - self.w3 * L_int
        else:
            x = (L_ext - L_sys - L_int) + self.non_linearity(self.w1 * L_ext - self.w2 * L_sys - self.w3 * L_int)

        return x

def init_weights(block_len, num_iter, device = torch.device('cpu'), init_type = 'ones', type = 'normal', non_linear=False):
    weight_dict = {}
    normal = {}
    interleaved = {}

    assert type in ['normal', 'normal_common', 'same_all', 'same_iteration', 'scale', 'scale_common', 'same_scale_iteration', 'same_scale', 'one_weight']

    if type == 'normal':
        for ii in range(num_iter):
            normal[ii] = Turbo_subnet(block_len, init_type, non_linear).to(device)
            interleaved[ii] = Turbo_subnet(block_len, init_type, non_linear).to(device)
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    if type == 'normal_common':
        for ii in range(num_iter):
            net = Turbo_subnet(block_len, init_type, non_linear).to(device)
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_all':
        net = Turbo_subnet(block_len, init_type, non_linear).to(device)
        for ii in range(num_iter):
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_iteration':
        normal_net = Turbo_subnet(block_len, init_type, non_linear).to(device)
        interleaved_net = Turbo_subnet(block_len, init_type, non_linear).to(device)

        for ii in range(num_iter):
            normal[ii] = normal_net
            interleaved[ii] = interleaved_net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'scale':
        for ii in range(num_iter):
            normal[ii] = Turbo_subnet(1, init_type, non_linear).to(device)
            interleaved[ii] = Turbo_subnet(1, init_type, non_linear).to(device)
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'scale_common':
        for ii in range(num_iter):
            net = Turbo_subnet(1, init_type, non_linear).to(device)
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_scale':
        net = Turbo_subnet(1, init_type, non_linear).to(device)
        for ii in range(num_iter):
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'same_scale_iteration':
        net_normal = Turbo_subnet(1, init_type, non_linear).to(device)
        net_interleaved = Turbo_subnet(1, init_type, non_linear).to(device)
        for ii in range(num_iter):
            normal[ii] = net_normal
            interleaved[ii] = net_interleaved
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    elif type == 'one_weight':
        net = Turbo_subnet(1, init_type, non_linear, one_weight = True).to(device)
        for ii in range(num_iter):
            normal[ii] = net
            interleaved[ii] = net
        weight_dict['normal'] = normal
        weight_dict['interleaved'] = interleaved

    return weight_dict

def turbonet_decode(weight_dict, received_llrs, trellis, number_iterations, interleaver, L_int = None, method = 'max_log_MAP', puncture = False, old = False):

    coded = received_llrs[:, :-4*trellis.total_memory]
    term = received_llrs[:, -4*trellis.total_memory:]
    if puncture:
        block_len = coded.shape[1]//2
        inds = torch.Tensor([1, 1, 0, 1, 0, 1]).repeat(block_len//2).byte()
        zero_inserted = torch.zeros(received_llrs.shape[0], 3*block_len, device = received_llrs.device)
        zero_inserted[:, inds] = coded
        coded = zero_inserted.float()
    sys_stream = coded[:, 0::3]
    non_sys_stream1 = coded[:, 1::3]
    non_sys_stream2 = coded[:, 2::3]

    term_sys1 = term[:, :2*trellis.total_memory][:, 0::2]
    term_nonsys1 = term[:, :2*trellis.total_memory][:, 1::2]
    term_sys2 = term[:, 2*trellis.total_memory:][:, 0::2]
    term_nonsys2 = term[:, 2*trellis.total_memory:][:, 1::2]

    sys_llrs = torch.cat((sys_stream, term_sys1), -1)
    non_sys_llrs1 = torch.cat((non_sys_stream1, term_nonsys1), -1)

    sys_stream_inter = interleaver.interleave(sys_stream)
    sys_llrs_inter = torch.cat((sys_stream_inter, term_sys2), -1)

    non_sys_llrs2 = torch.cat((non_sys_stream2, term_nonsys2), -1)
    sys_llr = sys_llrs

    if L_int is None:
        L_int = torch.zeros_like(sys_llrs).to(coded.device)

    L_int_1 = L_int

    for iteration in range(number_iterations):
        [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, trellis, L_int_1, method=method)

        if old:
            L_ext = L_ext_1
        else:
            L_ext = L_ext_1 - L_int_1 - sys_llr
        L_e_1 = L_ext_1[:, :sys_stream.shape[1]]
        L_1 = L_int_1[:, :sys_stream.shape[1]]

        L_int_2 = weight_dict['normal'][iteration](L_e_1, sys_llr[:, :sys_stream.shape[1]], L_1)
        L_int_2 = interleaver.interleave(L_int_2)
        L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)

        [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, trellis, L_int_2, method=method)

        L_e_2 = interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
        L_2 = interleaver.deinterleave(L_int_2[:, :sys_stream.shape[1]])
        L_int_1 = weight_dict['interleaved'][iteration](L_e_2, sys_llr[:, :sys_stream.shape[1]], L_2)

        L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)
    LLRs = L_ext + L_int_1 + sys_llr
    decoded_bits = (LLRs > 0).float()

    return LLRs, decoded_bits

def train(args, trellis1, trellis2, interleaver, device, loaded_weights = None):

    if loaded_weights is None:
        weight_dict = init_weights(args.block_len, args.turbonet_iters, device, args.init_type, args.decoding_type, args.non_linear)
    else:
        weight_dict = loaded_weights
        for ii in range(args.turbonet_iters):
            weight_dict['normal'][ii].to(device)
            weight_dict['interleaved'][ii].to(device)
    params = []
    for ii in range(args.turbonet_iters):
        params += list(weight_dict['normal'][ii].parameters())
        params += list(weight_dict['interleaved'][ii].parameters())

    criterion = nn.BCEWithLogitsLoss() if args.loss_type == 'BCE' else nn.MSELoss()
    optimizer = optim.Adam(params, lr = args.lr)

    sigma = snr_db2sigma(args.train_snr)
    noise_variance = sigma**2
    isi_filter = None

    noise_type = args.noise_type #if args.noise_type is not 'isi' else 'isi_1'
    if args.noise_type in ['isi', 'isi_perfect', 'isi_uncertain', 'epa', 'etu', 'eva']:
        if args.noise_type == 'epa':
            isi_filter = torch.Tensor([8.4568e-01, 3.2618e-01, 1.6513e-01, 8.0825e-02, 3.8788e-02,
            3.0822e-02, 7.4965e-02, 2.4612e-02, 9.1985e-03, 5.1362e-03, 2.8590e-03,
            1.4598e-03, 6.6643e-04, 2.4957e-04, 7.5514e-05]).float().to(device)
            noise_type = 'isi_perfect'
        if args.noise_type == 'eva':
            isi_filter = torch.Tensor([7.1455e-01, 3.1013e-01, 1.8716e-01, 5.9571e-02, 3.6789e-02,
            1.2863e-01, 9.9646e-02, 2.2660e-02, 1.1989e-02, 1.0632e-02, 1.9607e-02,
            1.8184e-01, 1.5879e-02, 1.2813e-02, 2.0223e-02, 3.8519e-02, 1.0080e-01,
            2.9634e-01, 5.6660e-02, 2.7016e-02, 1.4812e-02, 8.6525e-03, 5.9289e-03,
            5.8595e-03, 8.5137e-03, 1.5755e-02, 4.7614e-02, 6.4232e-02, 1.7690e-02,
            8.7245e-03, 4.6893e-03, 2.4575e-03, 1.1798e-03, 4.9426e-04, 2.8556e-04,
            4.8327e-04, 8.9919e-04, 1.7837e-03, 5.5005e-03, 6.8520e-03, 1.9425e-03,
            9.6152e-04, 5.1685e-04, 2.7053e-04, 1.2964e-04, 5.3155e-05, 1.6475e-05]).float().to(device)
            noise_type = 'isi_perfect'
        elif args.noise_type == 'etu':
            isi_filter = torch.Tensor([4.6599e-01, 3.8824e-01, 1.7547e-01, 1.9635e-01, 8.4110e-02,
            3.6070e-02, 5.8050e-02, 1.6326e-01, 3.5549e-01, 7.9869e-02, 3.8474e-02,
            2.0689e-02, 1.0934e-02, 5.3476e-03, 2.2747e-03, 7.7013e-04, 0.0000e+00,
            5.6003e-04, 1.9325e-03, 4.9110e-03, 1.0480e-02, 2.0250e-02, 3.7686e-02,
            7.4502e-02, 2.2443e-01, 3.0682e-01, 8.4079e-02, 4.1442e-02, 2.1998e-02,
            1.0862e-02, 3.7387e-03, 2.1157e-03, 7.2256e-03, 1.4741e-02, 3.0517e-02,
            1.3314e-01, 6.4063e-02, 2.3194e-02, 1.1887e-02, 6.3774e-03, 3.2680e-03,
            1.5022e-03, 5.7011e-04, 1.6832e-04]).float().to(device)
            noise_type = 'isi_perfect'
        else:
            isi_filter = torch.Tensor([1., 0.6]).float().to(device)

        equalizer = Equalizer(isi_filter, device)
        w = equalizer.get_equalizer(M = args.eq_M)
        print('Got equalizer')


    print("TRAINING")
    training_losses = []
    training_bers = []

    try:
        for step in range(args.num_steps):
            start = time.time()
            message_bits = torch.randint(0, 2, (args.batch_size, args.block_len), dtype=torch.float).to(device)
            coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
            noisy_coded = corrupt_signal(coded, sigma, noise_type, vv = args.vv, radar_power = args.radar_power, radar_prob = args.radar_prob, gamma = args.isi_gamma, isi_filter = isi_filter)

            #Turbonet decode

            if args.noise_type in ['isi', 'isi_perfect', 'isi_uncertain']:
                noisy_coded_received = noisy_coded.clone()
                noisy_coded = equalizer.equalize(noisy_coded_received)
            received_llrs = 2*noisy_coded/noise_variance

            if args.input == 'y':
                turbonet_llr, decoded_tn = turbonet_decode(weight_dict, noisy_coded, trellis1, args.turbonet_iters, interleaver, method = args.tn_bcjr, puncture = args.puncture)
            else:
                turbonet_llr, decoded_tn = turbonet_decode(weight_dict, received_llrs, trellis1, args.turbonet_iters, interleaver, method = args.tn_bcjr, puncture = args.puncture)


            if args.target == 'LLR':
                #Turbo decode
                log_map_llr, _ = turbo_decode(received_llrs, trellis1, args.turbo_iters, interleaver, method='log_MAP', puncture = args.puncture)
                loss = criterion(turbonet_llr, log_map_llr)
            elif args.target == 'gt':
                if args.loss_type == 'BCE':
                    loss = criterion(turbonet_llr[:, :-trellis1.total_memory], message_bits)
                elif args.loss_type == 'MSE':
                    loss = criterion(torch.tanh(turbonet_llr[:, :-trellis1.total_memory]/2.), 2*message_bits-1)
            ber = errors_ber(message_bits, decoded_tn[:, :-trellis1.total_memory])

            training_losses.append(loss.item())
            training_bers.append(ber)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1)%10 == 0:
                print('Step : {}, Loss = {:.5f}, BER = {:.5f}, {:.2f} seconds, ID: {}'.format(step+1, loss, ber, time.time() - start, args.id))

            if (step+1)%args.save_every == 0 or step==0:
                torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
                torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(step+1))))
            if (step+1)%10 == 0:
                plt.figure()
                plt.plot(training_losses)
                plt.plot(moving_average(training_losses, n=10))
                plt.savefig(os.path.join(args.save_path, 'training_losses.png'))
                plt.close()

                plt.figure()
                plt.plot(training_losses)
                plt.plot(moving_average(training_losses, n=10))
                plt.yscale('log')
                plt.savefig(os.path.join(args.save_path, 'training_losses_log.png'))
                plt.close()

                plt.figure()
                plt.plot(training_bers)
                plt.plot(moving_average(training_bers, n=10))
                plt.savefig(os.path.join(args.save_path, 'training_bers.png'))
                plt.close()

                plt.figure()
                plt.plot(training_bers)
                plt.plot(moving_average(training_bers, n=10))
                plt.yscale('log')
                plt.savefig(os.path.join(args.save_path, 'training_bers_log.png'))
                plt.close()

                with open(os.path.join(args.save_path, 'values_training.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    write = csv.writer(f)

                    write.writerow(list(range(1, step+1)))
                    write.writerow(training_losses)
                    write.writerow(training_bers)

        return weight_dict, training_losses, training_bers, step+1

    except KeyboardInterrupt:
        print("Exited")

        torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
        torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(step+1))))
        with open(os.path.join(args.save_path, 'values_training.csv'), 'w') as f:

             # using csv.writer method from CSV package
             write = csv.writer(f)
             write.writerow(list(range(1, step+1)))
             write.writerow(training_losses)
             write.writerow(training_bers)

        return weight_dict, training_losses, training_bers, step+1

def test(args, weight_d, trellis1, trellis2, interleaver, device, only_tn = False):

    if args.snr_points == 1 and args.test_snr_start == args.test_snr_end:
        snr_range = [args.test_snr_start]
    else:
        snrs_interval = (args.test_snr_end - args.test_snr_start) * 1.0 / (args.snr_points-1)
        snr_range = [snrs_interval * item + args.test_snr_start for item in range(args.snr_points)]

    num_batches = args.test_size // args.test_batch_size
    isi_filter = None
    noise_type = args.noise_type #if args.noise_type is not 'isi' else 'isi_1'
    if args.noise_type in ['isi', 'isi_perfect', 'isi_uncertain', 'epa', 'etu', 'eva']:
        if args.noise_type == 'epa':
            isi_filter = torch.Tensor([8.4568e-01, 3.2618e-01, 1.6513e-01, 8.0825e-02, 3.8788e-02,
            3.0822e-02, 7.4965e-02, 2.4612e-02, 9.1985e-03, 5.1362e-03, 2.8590e-03,
            1.4598e-03, 6.6643e-04, 2.4957e-04, 7.5514e-05]).float().to(device)
            noise_type = 'isi_perfect'
        if args.noise_type == 'eva':
            isi_filter = torch.Tensor([7.1455e-01, 3.1013e-01, 1.8716e-01, 5.9571e-02, 3.6789e-02,
            1.2863e-01, 9.9646e-02, 2.2660e-02, 1.1989e-02, 1.0632e-02, 1.9607e-02,
            1.8184e-01, 1.5879e-02, 1.2813e-02, 2.0223e-02, 3.8519e-02, 1.0080e-01,
            2.9634e-01, 5.6660e-02, 2.7016e-02, 1.4812e-02, 8.6525e-03, 5.9289e-03,
            5.8595e-03, 8.5137e-03, 1.5755e-02, 4.7614e-02, 6.4232e-02, 1.7690e-02,
            8.7245e-03, 4.6893e-03, 2.4575e-03, 1.1798e-03, 4.9426e-04, 2.8556e-04,
            4.8327e-04, 8.9919e-04, 1.7837e-03, 5.5005e-03, 6.8520e-03, 1.9425e-03,
            9.6152e-04, 5.1685e-04, 2.7053e-04, 1.2964e-04, 5.3155e-05, 1.6475e-05]).float().to(device)
            noise_type = 'isi_perfect'
        elif args.noise_type == 'etu':
            isi_filter = torch.Tensor([4.6599e-01, 3.8824e-01, 1.7547e-01, 1.9635e-01, 8.4110e-02,
            3.6070e-02, 5.8050e-02, 1.6326e-01, 3.5549e-01, 7.9869e-02, 3.8474e-02,
            2.0689e-02, 1.0934e-02, 5.3476e-03, 2.2747e-03, 7.7013e-04, 0.0000e+00,
            5.6003e-04, 1.9325e-03, 4.9110e-03, 1.0480e-02, 2.0250e-02, 3.7686e-02,
            7.4502e-02, 2.2443e-01, 3.0682e-01, 8.4079e-02, 4.1442e-02, 2.1998e-02,
            1.0862e-02, 3.7387e-03, 2.1157e-03, 7.2256e-03, 1.4741e-02, 3.0517e-02,
            1.3314e-01, 6.4063e-02, 2.3194e-02, 1.1887e-02, 6.3774e-03, 3.2680e-03,
            1.5022e-03, 5.7011e-04, 1.6832e-04]).float().to(device)
            noise_type = 'isi_perfect'
        else:
            isi_filter = torch.Tensor([1., 0.6]).float().to(device)
        equalizer = Equalizer(isi_filter, device)
        w = equalizer.get_equalizer(M = args.eq_M)
        print('Got equalizer')

    bers_ml = []
    blers_ml = []
    bers_l = []
    blers_l = []
    bers_tn = []
    blers_tn = []
    print("TESTING")

    for ii in range(num_batches):
        message_bits = torch.randint(0, 2, (args.test_batch_size, args.block_len), dtype=torch.float).to(device)
        coded = turbo_encode(message_bits, trellis1, trellis2, interleaver, puncture = args.puncture).to(device)
        for k, snr in tqdm(enumerate(snr_range)):
            sigma = snr_db2sigma(snr)
            noise_variance = sigma**2

            # noise_type = args.noise_type #if args.noise_type is not 'isi' else 'isi_1'
            noisy_coded = corrupt_signal(coded, sigma, noise_type, vv = args.vv, radar_power = args.radar_power, radar_prob = args.radar_prob, gamma = args.isi_gamma, isi_filter = isi_filter)

            if args.noise_type in ['isi', 'isi_perfect', 'isi_uncertain']:
                noisy_coded_received = noisy_coded.clone()
                noisy_coded = equalizer.equalize(noisy_coded_received)

            received_llrs = 2*noisy_coded/noise_variance

            if not only_tn:
                # Turbo decode
                ml_llrs, decoded_ml = turbo_decode(received_llrs, trellis1, args.turbonet_iters,
                                             interleaver, method='max_log_MAP', puncture = args.puncture)
                ber_maxlog = errors_ber(message_bits, decoded_ml[:, :-trellis1.total_memory])
                bler_maxlog = errors_bler(message_bits, decoded_ml[:, :-trellis1.total_memory])

                if ii == 0:
                    bers_ml.append(ber_maxlog/num_batches)
                    blers_ml.append(bler_maxlog/num_batches)
                else:
                    bers_ml[k] += ber_maxlog/num_batches
                    blers_ml[k] += bler_maxlog/num_batches

                l_llrs, decoded_l = turbo_decode(received_llrs, trellis1, args.turbo_iters,
                                            interleaver, method='log_MAP', puncture = args.puncture)
                ber_log = errors_ber(message_bits, decoded_l[:, :-trellis1.total_memory])
                bler_log = errors_bler(message_bits, decoded_l[:, :-trellis1.total_memory])

                if ii == 0:
                    bers_l.append(ber_log/num_batches)
                    blers_l.append(bler_log/num_batches)
                else:
                    bers_l[k] += ber_log/num_batches
                    blers_l[k] += bler_log/num_batches

            # Turbonet decode
            if args.input == 'y':
                tn_llrs, decoded_tn = turbonet_decode(weight_d, noisy_coded, trellis1, args.turbonet_iters, interleaver, method = args.tn_bcjr, puncture = args.puncture)
            else:
                tn_llrs, decoded_tn = turbonet_decode(weight_d, received_llrs, trellis1, args.turbonet_iters, interleaver, method = args.tn_bcjr, old = args.old, puncture = args.puncture)

            ber_turbonet = errors_ber(message_bits, decoded_tn[:, :-trellis1.total_memory])
            bler_turbonet = errors_bler(message_bits, decoded_tn[:, :-trellis1.total_memory])

            if ii == 0:
                bers_tn.append(ber_turbonet/num_batches)
                blers_tn.append(bler_turbonet/num_batches)
            else:
                bers_tn[k] += ber_turbonet/num_batches
                blers_tn[k] += bler_turbonet/num_batches

    return snr_range, bers_ml, bers_l, bers_tn, blers_ml, blers_l, blers_tn


def train_real(args, trellis1, trellis2, interleaver, device, train_loader, loaded_weights = None):

    if loaded_weights is None:
        weight_dict = init_weights(args.block_len, args.turbonet_iters, device, args.init_type, args.decoding_type, args.non_linear)
    else:
        weight_dict = loaded_weights
        for ii in range(args.turbonet_iters):
            weight_dict['normal'][ii].to(device)
            weight_dict['interleaved'][ii].to(device)
    params = []
    for ii in range(args.turbonet_iters):
        params += list(weight_dict['normal'][ii].parameters())
        params += list(weight_dict['interleaved'][ii].parameters())

    criterion = nn.BCEWithLogitsLoss() if args.loss_type == 'BCE' else nn.MSELoss()
    optimizer = optim.Adam(params, lr = args.lr)

    print("TRAINING")
    training_losses = []
    training_bers = []

    try:
        for step, (received_llrs, message_bits) in enumerate(tqdm(train_loader)):
            start = time.time()
            received_llrs = received_llrs.to(device)
            message_bits = message_bits.to(device)
            turbonet_llr, decoded_tn = turbonet_decode(weight_dict, received_llrs, trellis1, args.turbonet_iters, interleaver, method = args.tn_bcjr, puncture = args.puncture)

            if args.target == 'LLR':
                #Turbo decode
                log_map_llr, _ = turbo_decode(received_llrs, trellis1, args.turbo_iters, interleaver, method='log_MAP', puncture = args.puncture)
                loss = criterion(turbonet_llr, log_map_llr)
            elif args.target == 'gt':
                if args.loss_type == 'BCE':
                    loss = criterion(turbonet_llr[:, :-trellis1.total_memory], message_bits)
                elif args.loss_type == 'MSE':
                    loss = criterion(torch.tanh(turbonet_llr[:, :-trellis1.total_memory]/2.), 2*message_bits-1)
            ber = errors_ber(message_bits, decoded_tn[:, :-trellis1.total_memory])

            training_losses.append(loss.item())
            training_bers.append(ber)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Step : {}, Loss = {}, BER = {}, {} seconds'.format(step+1, loss, ber, time.time() - start))

            if (step+1)%args.save_every == 0 or step==9:
                torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
                torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(step+1))))
            if (step+1)%10 == 0:
                plt.figure()
                plt.plot(training_losses)
                plt.plot(moving_average(training_losses, n=10))
                plt.savefig(os.path.join(args.save_path, 'training_losses.png'))
                plt.close()

                plt.figure()
                plt.plot(training_losses)
                plt.plot(moving_average(training_losses, n=10))
                plt.yscale('log')
                plt.savefig(os.path.join(args.save_path, 'training_losses_log.png'))
                plt.close()

                plt.figure()
                plt.plot(training_bers)
                plt.plot(moving_average(training_bers, n=10))
                plt.savefig(os.path.join(args.save_path, 'training_bers.png'))
                plt.close()

                plt.figure()
                plt.plot(training_bers)
                plt.plot(moving_average(training_bers, n=10))
                plt.yscale('log')
                plt.savefig(os.path.join(args.save_path, 'training_bers_log.png'))
                plt.close()
        return weight_dict, training_losses, training_bers, step+1

    except KeyboardInterrupt:
        print("Exited")

        torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights.pt'))
        torch.save({'weights': weight_dict, 'args': args, 'steps': step+1, 'p_array':interleaver.p_array}, os.path.join(args.save_path, 'models/weights_{}.pt'.format(int(step+1))))
        return weight_dict, training_losses, training_bers, step+1

def test_real(args, weight_d, trellis1, trellis2, interleaver, device, test_loader):

    bers_ml = 0.
    blers_ml = 0.
    bers_l = 0.
    blers_l = 0.
    bers_tn = 0.
    blers_tn = 0.

    num_batches = len(test_loader)
    print("TESTING")
    for idx, (received_llrs, message_bits) in enumerate(tqdm(test_loader)):

        received_llrs = received_llrs.to(device)
        message_bits = message_bits.to(device)
        # Turbo decode
        _, decoded_ml = turbo_decode(received_llrs, trellis1, args.turbonet_iters,
                                     interleaver, method='max_log_MAP', puncture = args.puncture)
        ber_maxlog = errors_ber(message_bits, decoded_ml[:, :-trellis1.total_memory])
        bler_maxlog = errors_bler(message_bits, decoded_ml[:, :-trellis1.total_memory])
        bers_ml += ber_maxlog/num_batches
        blers_ml += bler_maxlog/num_batches

        _, decoded_l = turbo_decode(received_llrs, trellis1, args.turbo_iters,
                                    interleaver, method='log_MAP', puncture = args.puncture)
        ber_log = errors_ber(message_bits, decoded_l[:, :-trellis1.total_memory])
        bler_log = errors_bler(message_bits, decoded_l[:, :-trellis1.total_memory])
        bers_l += ber_log/num_batches
        blers_l += bler_log/num_batches

        # Turbonet decode
        _, decoded_tn = turbonet_decode(weight_d, received_llrs, trellis1, args.turbonet_iters, interleaver, method = args.tn_bcjr, old = args.old, puncture = args.puncture)
        ber_turbonet = errors_ber(message_bits, decoded_tn[:, :-trellis1.total_memory])
        bler_turbonet = errors_bler(message_bits, decoded_tn[:, :-trellis1.total_memory])
        bers_tn += ber_turbonet/num_batches
        blers_tn += bler_turbonet/num_batches

    return bers_ml, bers_l, bers_tn, blers_ml, blers_l, blers_tn
