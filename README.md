# DeepIC
Source code for the paper: DeepIC+: Learning Codes for Interference Channels

Abstract: A two-user interference channel is a canonical model for multiple one-to-one communications, where two transmitters wish to communicate with their receivers via a shared medium, examples of which include pairs of base stations and handsets near the cell boundary that suffer from interference. Practical codes and the fundamental limit of communications are unknown for interference channels as mathematical analysis becomes intractable. Hence, simple heuristic coding schemes are used in practice to mitigate the interference, e.g.,
time division, treating interference as noise, and successive interference cancellation. These schemes are nearly optimal for extreme cases: when the interference is strong or weak. However, there is no optimality guarantee for channels with moderate interference. Designing reliable codes for channels with moderate interference is a long-standing open problem. Here we combine deep learning and network information theory to overcome the limitation on the tractability of analysis and construct finite-blocklength coding schemes for channels with various interference levels. 
We show that neural codes, carefully designed and trained using network information theoretic insight, can achieve several orders of reliability improvement for channels with moderate interference. Furthermore, we present the interpretation of the learned codes based on the codeword distance and the Centered Kernel Alignment (CKA) analysis.  

For training and testing, run the file main_pretrain.py. The code first initializes the encoders and decoders with models pretrained on a point-to-point AWGN channel, and then finetunes the overall model on the interference channel. Once training is over, the performance is evaluated over a range of SNRs.

For settings such as training/testing SNRs, batch size, learning rate, please refer to the file get_args.py.

A pretrained point-to-point model trained with blocklength = 40 is provided (under p2p_models_bl40).
