# Deep Cascade Generation on Point Sets

Kaiqi Wang, Ke Chen*, Kui Jia &nbsp; &nbsp;
IJCAI 2019

[[paper](https://www.ijcai.org/proceedings/2019/0517.pdf)] | [[project page](https://wkqscut.github.io/DCGNet/)]

This implementation uses [Pytorch](http://pytorch.org/).

## Installation

```shell
git clone https://github.com/wkqscut/DCGNet.git
cd DCGNet
## Create python env with relevant packages
conda create --name dcg python=3.7
conda activate dcg
pip install -U pip
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch  # cudatoolkit=10.0 for cuda10
```

Tested on pytorch >= 1.0 and python3.

## Build

```shell
## Build chamfer distance
conda activate dcg
cd ./extension
python setup.py install
cd ../
```

## Download
### Dataset
We used the rendered imaged from [3d-R2N2](https://github.com/chrischoy/3D-R2N2), and the groundtruth 3D point clouds sampled from [ShapeNet](https://www.shapenet.org/).
* [The rendered images (*.png)](https://cloud.enpc.fr/s/S6TCx1QJzviNHq0) go in ```./data/ShapeNetRendering```
* [The ShapeNetPoints (*.ply) datasets](https://cloud.enpc.fr/s/j2ECcKleA1IKNzk) go in ```./data/ShapeNetPoints```

### Pretrained models
* [The pretrained models (*.pth)](https://drive.google.com/open?id=1VjpPsbDepy90VBJCM2_PVx1_lXJ_Dzsi) should be placed in ```./trained_models/```

```shell
## unzip the dataset and Pretrained models using the scripts
bash ./trained_models/unzip_models_dataset.sh
```


## Run code

### Demo
* demo code for DCGNet
```shell
bash ./scripts/demo.sh
```


### Training

* train the DCGNet for Point Set AutoEncoding:
```shell
bash ./scripts/train_svr_dcg.sh
```

* train the DCGNet for Point Set Reconstruction from a Single Image:
```shell
bash ./scripts/train_svr_dcg.sh
```


### Inference

* test the DCGNet for Point Set AutoEncoding:
```shell
bash ./scripts/test_svr_dcg.sh
```

* test the DCGNet for Point Set Reconstruction from a Single Image:
```shell
bash ./scripts/test_svr_dcg.sh
```

## Citing this work
If you find this code useful for your research, please consider citing the following paper:

    @inproceedings{ijcai2019-517,
      title     = {Deep Cascade Generation on Point Sets},
      author    = {Wang, Kaiqi and Chen, Ke and Jia, Kui},
      booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
                   Artificial Intelligence, {IJCAI-19}},
      publisher = {International Joint Conferences on Artificial Intelligence Organization},
      pages     = {3726--3732},
      year      = {2019},
      month     = {7},
      doi       = {10.24963/ijcai.2019/517},
      url       = {https://doi.org/10.24963/ijcai.2019/517},
    }

## Acknowledgements
This work is supported in part by the Program for Guangdong Introducing Innovative and Enterpreneurial Teams (Grant No.: 2017ZT07X183), the National Natural Science Foundation of China (Grant No.: 61771201), and the Program of the Construction of Talented Personnel by the South China University of Technology (Grant No.: D6192110).
