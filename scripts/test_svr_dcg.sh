#!/usr/bin/env bash
set -x
set -e

synsets=('allclass')  # 'car'

# change the gpu number
export CUDA_VISIBLE_DEVICES=0
source activate dcg

nb_primitives1=5
nb_primitives2=5
python ./inference/run_SVR_DCGNet.py --classname $synsets --nb_primitives1 $nb_primitives1 --nb_primitives2 $nb_primitives2
--model './trained_models/svr_dcg.pth'