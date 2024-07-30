import numpy as np
import torch
import torch.nn.functional as F

def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of ints
        Array containing the binary representation of the input decimal.

    """

    binary_string = bin(in_number)
    length = len(binary_string)
    bitarray = np.zeros(bit_width, 'int')
    for i in range(length-2):
        bitarray[bit_width-i-1] = int(binary_string[length-i-1])

    return bitarray

def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i]*pow(2, len(in_bitarray)-1-i)

    return number

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def errors_ber(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return res.item()


def errors_bler(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate

def corrupt_signal(input_signal, sigma = 1.0, noise_type = 'awgn', vv =5.0, radar_power = 20.0, radar_prob = 5e-2, gamma = 1, isi_filter = None):

    data_shape = input_signal.shape  # input_signal has to be a numpy array.
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist', 'isi', 'eva', 'etu', 'epa', 'isi', 'isi_1', 'isi_perfect', 'isi_uncertain'], "Invalid noise type"

    if noise_type == 'awgn':
        noise = sigma * torch.randn_like(input_signal) # Define noise
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 'fading':
        fading_h = torch.sqrt(torch.randn_like(input_signal)**2 +  torch.randn_like(input_signal)**2)/np.sqrt(3.14/2.0)
        noise = sigma * torch.randn_like(input_signal) # Define noise
        corrupted_signal = fading_h *(2.0*input_signal-1.0) + noise

    elif noise_type == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], data_shape,
                                       p=[1 - radar_prob, radar_prob])

        corrupted_signal = radar_power* np.random.standard_normal( size = data_shape ) * add_pos
        noise = sigma * torch.randn_like(input_signal) +\
                    torch.from_numpy(corrupted_signal).type(torch.FloatTensor).to(input_signal.device)
        corrupted_signal = 2.0*input_signal-1.0 + noise

    elif noise_type == 't-dist':
        noise = sigma * np.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)
        corrupted_signal = 2.0*input_signal-1.0 + torch.from_numpy(noise).type(torch.FloatTensor).to(input_signal.device)

    elif noise_type == 'isi':
        filter_len = 4

        h = torch.exp(-gamma*torch.arange(filter_len, dtype=torch.float)).to(input_signal.device)
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(h, [0,]).float().view(1, 1, -1), padding = h.shape[0] - 1).squeeze()[:input_reshaped.shape[0]]
        corrupted_signal = torch.reshape(out, input_signal.shape) + sigma * torch.randn_like(input_signal)


    elif noise_type == 'eva':
        eva_filter = torch.tensor([8.5192e-04, 2.7762e-03, 6.6923e-03, 1.3914e-02, 2.6517e-02, 4.9323e-02,
        9.9711e-02, 7.1455e-01, 3.1013e-01, 1.8716e-01, 5.9571e-02, 3.6789e-02,
        1.2863e-01, 9.9646e-02, 2.2660e-02, 1.1989e-02, 1.0632e-02, 1.9607e-02,
        1.8184e-01, 1.5879e-02, 1.2813e-02, 2.0223e-02, 3.8519e-02, 1.0080e-01,
        2.9634e-01, 5.6660e-02, 2.7016e-02, 1.4812e-02, 8.6525e-03, 5.9289e-03,
        5.8595e-03, 8.5137e-03, 1.5755e-02, 4.7614e-02, 6.4232e-02, 1.7690e-02,
        8.7245e-03, 4.6893e-03, 2.4575e-03, 1.1798e-03, 4.9426e-04, 2.8556e-04,
        4.8327e-04, 8.9919e-04, 1.7837e-03, 5.5005e-03, 6.8520e-03, 1.9425e-03,
        9.6152e-04, 5.1685e-04, 2.7053e-04, 1.2964e-04, 5.3155e-05, 1.6475e-05,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00], dtype=torch.float64).to(input_signal.device)

        peak = torch.argmax(eva_filter)
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(eva_filter, [0,]).float().view(1, 1, -1), padding = eva_filter.shape[0] - 1).squeeze()[peak:peak+input_reshaped.shape[0]]
        corrupted_signal = torch.reshape(out, input_signal.shape) + sigma * torch.randn_like(input_signal)

    elif noise_type == 'etu':
        etu_filter = torch.tensor([3.4472e-04, 1.0813e-03, 2.9910e-03, 6.7699e-03, 1.3555e-02, 2.5591e-02,
        4.9492e-02, 4.6599e-01, 3.8824e-01, 1.7547e-01, 1.9635e-01, 8.4110e-02,
        3.6070e-02, 5.8050e-02, 1.6326e-01, 3.5549e-01, 7.9869e-02, 3.8474e-02,
        2.0689e-02, 1.0934e-02, 5.3476e-03, 2.2747e-03, 7.7013e-04, 0.0000e+00,
        5.6003e-04, 1.9325e-03, 4.9110e-03, 1.0480e-02, 2.0250e-02, 3.7686e-02,
        7.4502e-02, 2.2443e-01, 3.0682e-01, 8.4079e-02, 4.1442e-02, 2.1998e-02,
        1.0862e-02, 3.7387e-03, 2.1157e-03, 7.2256e-03, 1.4741e-02, 3.0517e-02,
        1.3314e-01, 6.4063e-02, 2.3194e-02, 1.1887e-02, 6.3774e-03, 3.2680e-03,
        1.5022e-03, 5.7011e-04, 1.6832e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00], dtype=torch.float64).to(input_signal.device)

        peak = torch.argmax(etu_filter)
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(etu_filter, [0,]).float().view(1, 1, -1), padding = etu_filter.shape[0] - 1).squeeze()[peak:peak+input_reshaped.shape[0]]
        corrupted_signal = torch.reshape(out, input_signal.shape) + sigma * torch.randn_like(input_signal)

    elif noise_type == 'epa':
        epa_filter = torch.tensor([1.0415e-03, 3.4076e-03, 8.3427e-03, 1.7438e-02, 3.3339e-02, 6.1985e-02,
        1.2468e-01, 8.4568e-01, 3.2618e-01, 1.6513e-01, 8.0825e-02, 3.8788e-02,
        3.0822e-02, 7.4965e-02, 2.4612e-02, 9.1985e-03, 5.1362e-03, 2.8590e-03,
        1.4598e-03, 6.6643e-04, 2.4957e-04, 7.5514e-05, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00], dtype=torch.float64).to(input_signal.device)

        peak = torch.argmax(epa_filter)
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(epa_filter, [0,]).float().view(1, 1, -1), padding = epa_filter.shape[0] - 1).squeeze()[peak:peak+input_reshaped.shape[0]]
        corrupted_signal = torch.reshape(out, input_signal.shape)+ sigma * torch.randn_like(input_signal)

    elif noise_type == 'isi_perfect':
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(isi_filter, [0,]).float().view(1, 1, -1), padding = isi_filter.shape[0] - 1).squeeze()[:input_reshaped.shape[0]]
        corrupted_signal = torch.reshape(out, input_signal.shape) + sigma * torch.randn_like(input_signal)

    elif noise_type == 'isi_uncertain':
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        a = 0.9+(torch.rand_like(input_reshaped) * 0.2)
        b = 0.5+(torch.rand_like(input_reshaped) * 0.2)
        out = a*input_reshaped + b*torch.cat([torch.zeros(1, device = input_reshaped.device), input_reshaped[:-1]])

        corrupted_signal = torch.reshape(out, input_signal.shape) + sigma * torch.randn_like(input_signal)

    return corrupted_signal

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class Equalizer():
    def __init__(self, isi_filter, device):
        self.w = None
        self.isi_filter = isi_filter
        self.device = device

    def get_equalizer(self, M=50):
        x = torch.randn(100000)
        y = F.conv1d(x.view(1, 1, -1), torch.flip(self.isi_filter.cpu(), [0,]).float().view(1, 1, -1), padding = self.isi_filter.shape[0] - 1).squeeze()[:x.shape[0]]
        e, w = self.Linear_LMS(y, x, M)
        self.w = w.to(self.device)
        self.filter_len = M

        return w
    # def Linear_LMS(x, y, M, step = 0.003, num_epochs = 1):
    #
    #     # input numpy for now
    #
    #     N = len(x) - M + 1
    #     # Initialization
    #     f = np.zeros(N)  # Filter output
    #     e = np.zeros(N)  # Error signal
    #     w = np.zeros(M)  # Initialise equaliser
    #     # Equalise
    #     for epoch in range(num_epochs):
    #         f = np.zeros(N)
    #         e = np.zeros(N)
    #         for n in range(N):
    #             xn = np.flipud(x[n : n + M])  #
    #             f[n] = np.dot(xn, w)
    #             e[n] = y[n + M - 1] - f[n]
    #             w = w + step * xn * e[n]
    #             #print(w)
    #         print(epoch, e[-1])
    #
    #     self.w = w
    #     self.filter_len = self.w.shape[0]
    #
    #     return e, w

    def Linear_LMS(self, y, x, M, step = 0.003):

        # input numpy for now

        N = len(y) - M + 1
        # Initialization
        f = torch.zeros(N)  # Filter output
        e = torch.zeros(N)  # Error signal
        w = torch.zeros(M)  # Initialise equaliser

        # Equalise
        for n in range(N):
            yn = torch.flip(y[n : n + M], [0])  #
            f[n] = torch.dot(yn, w)
            e[n] = x[n + M - 1] - f[n]
            w = w + step * yn * e[n]
            #print(w)
        return e, w

    def equalize(self, input_signal):
        input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
        out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(self.w, [0,]).float().view(1, 1, -1), padding = self.filter_len - 1).squeeze()[:input_reshaped.shape[0]]
        x_hat = torch.reshape(out, input_signal.shape)

        return x_hat
