
# Data

|ATSA-Dataset|Train |Test|
| :----:| :----: | :----:|
| restaurant  |   3693  |  1134   |
|   laptop   |   2358  |   654  |

|ACSA-Dataset|Train |Test|
| :----:| :----: | :----:|
|   restaurant-2014   |  3713   |   1025  |
|   restaurant-large   |  4665   |  2426   |

# Requirements
tensorboardX-2.1  

sacremoses 0.0.43  

torchtext 0.5.0  

pytorch 1.4+  


# Run


Download the glove word vectors and change the path in train.py in the line 34

If you run the code in Linux, you should change the path of train.py in line 102, from r'\\' to r'/'

## ATSA Task
python train.py -atsa  -lr 5e-3 -batch_size 32 -model gcae_atsa -atsa_data rest -epochs 6

python train.py -atsa  -lr 5e-3 -batch_size 32 -model gcae_atsa -atsa_data laptop -epochs 5

## ACSA Task

python train.py -lr 1e-2 -batch_size 32 -model gcae_acsa -acsa_data 2014  -epochs 5  

python train.py -lr 1e-2 -batch_size 32 -model gcae_acsa -acsa_data large  -epochs 13  



# Experiment Result


# Reference

- https://github.com/wxue004cs/GCAE (authors' code)
- https://github.com/songyouwei/ABSA-PyTorch










