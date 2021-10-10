# Introduction  

A reproduce about paper "Aspect Based Sentiment Analysis with Gated Convolutional Networks"(ACL 2018) using PyTorch. Welcome to discuss!  


# Requirements  

tensorboardX-2.1  

sacremoses 0.0.43  

torchtext 0.5.0  

pytorch 1.9.0 


# Run


Download the glove word vectors and change the path in train.py in the line 33

If you run the code in Linux, you should change the path of train.py in line 102, from r"\\" to r"/"

## ATSA Task
python train.py -atsa  -lr 5e-3 -batch_size 32 -model gcae_atsa -atsa_data rest -epochs 6

python train.py -atsa  -lr 5e-3 -batch_size 32 -model gcae_atsa -atsa_data laptop -epochs 5

## ACSA Task

python train.py -lr 1e-2 -batch_size 32 -model gcae_acsa -acsa_data 2014  -epochs 5  

python train.py -lr 1e-2 -batch_size 32 -model gcae_acsa -acsa_data large  -epochs 13  



# Reproduction-Experiment Result  

ACSA Task:

![ACSA Task](https://pic1.zhimg.com/80/v2-d7d2e79607d784a1826b1cc628ed09bf_720w.png)  

ATSA Task:

![ATSA Task](https://pic2.zhimg.com/80/v2-4dab74d75cd8d0ab96127e9b2db9f747_720w.png)



# Reference

- https://github.com/wxue004cs/GCAE (authors' code)
- https://github.com/songyouwei/ABSA-PyTorch










