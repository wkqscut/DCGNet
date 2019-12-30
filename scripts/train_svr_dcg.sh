#!/usr/bin/env bash
set -x
set -e

synsets=('allclass')  # 'car'
LOSS_PATH="./loss"

# change the gpu number
export CUDA_VISIBLE_DEVICES=0
source activate dcg

echo $synsets
mkdir -p $LOSS_PATH/$synsets
nb_primitives1=5
nb_primitives2=5
env=loss/$synsets/svr_dcg.`date +'%Y-%m-%d_%H-%M-%S'`
python ./training/train_SVR_DCGNet.py --classname $synsets --lr 0.001 --lr_decay1 300 --lr_decay2 400 \
--nepoch 420 --nb_primitives1 $nb_primitives1 --nb_primitives2 $nb_primitives2 --env $env |& tee ${env}.txt
