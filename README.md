# DeepIC
Coding for Interference Channels via Deep Learning

For training, run the file main_pretrain.py. The code first initializes the encoders and decoders with models pretrained on a point-to-point AWGN channel, and then finetunes the overall model on the interference channel.

A pretrained point-to-point model trained with blocklength = 40 is provided (under p2p_models_bl40).
