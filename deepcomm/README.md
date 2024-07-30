# deepcomm

This repository includes:
- A PyTorch implementation of Turbo encoding and decoding.
- Neural Turbo decoder inspired by the paper ["Model-Driven DNN Decoder for Turbo Codes: Design, Simulation and Experimental Results"](https://arxiv.org/abs/2006.08896)

All Python libraries required can be installed using:
```
pip install -r requirements.txt
```

## Training Neural Turbo decoder

Models are saved (with frequency specified by args.save_every) at Results/args.id/models/weights_{step_number}.pt

```
python main.py --batch_size 500 --block_len 40 --target gt --loss_type BCE --init_type ones --num_steps 1000 --turbonet_iters 3 --turbo_iters 6 --train_snr -1 --lr 0.0008 --noise_type awgn --gpu 0 --id *string_of_your_choice* 
```

## Training Neural Turbo decoder from saved model checkpoint

```
python main.py --batch_size 500 --block_len 40 --target gt --loss_type BCE --init_type ones --num_steps 1000 --turbonet_iters 3 --turbo_iters 6 --train_snr -1 --lr 0.0008 --noise_type awgn --gpu 0 --id *string_of_your_choice* --load_model_train *path to .pt file to initialize from*
```

## Training Neural Turbo decoder on OTA data

Models are saved (with frequency specified by args.save_every) at Results/args.id/models/weights_{step_number}.pt

```
python main.py --batch_size 500 --block_len 40 --target gt --loss_type BCE --init_type ones --num_steps 1000 --turbonet_iters 3 --turbo_iters 6 --train_snr -1 --lr 0.0008 --noise_type awgn --gpu 0 --id *string_of_your_choice* --train_data *path to train data* --test_data *path to test data*


```

## Testing Neural Turbo decoder

Tests the final model checkpoint at Results/args.id/models/weights.pt

```
python main.py --test_batch_size 10000 --block_len 40 --turbonet_iters 3 --turbo_iters 6 --noise_type awgn --gpu 0 --id *id of trained model* --test
```

## Testing Neural Turbo decoder at step_number

Tests the final model checkpoint at Results/args.id/models/weights_{step_number}.pt

```
python main.py --test_batch_size 10000 --block_len 40 --turbonet_iters 3 --turbo_iters 6 --noise_type awgn --gpu 0 --id *id of trained model* --test --load_model_step *step_number*
```

## Testing Neural Turbo decoder on OTA data 

```
python main.py --test_batch_size 10000 --block_len 40 --turbonet_iters 3 --turbo_iters 6 --noise_type awgn --gpu 0 --id *id of trained model* --test --load_model_step *step_number* --test_data *path to test data*
```

## Description of functions

- [convcode.py/conv_encode](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/convcode.py) : Convolutional code encoding
- [turbo.py/bcjr_decode](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/turbo.py) : BCJR (MAP) decoding of convolutional code
- [turbo.py/turbo_encode](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/turbo.py) : Turbo code encoding
- [turbo.py/turbo_decode](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/turbo.py) : Turbo decoder
- [turbonet.py/turbonet_decode](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/turbonet.py) : Neural turbo decode
- [turbonet.py/train](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/turbonet.py) : Train neural turbo decoder
- [turbonet.py/test](https://github.com/hebbarashwin/deepcomm/blob/147e30f3ce1f0242f92721b32a9d757aad6291a1/turbonet.py) : Test neural turbo decoder

