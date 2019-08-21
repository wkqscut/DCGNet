"""
DCGNet for Point Set Reconstruction from a Single Image.
@run_SVR_DCGNet.py
"""

from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import sys
sys.path.append('./auxiliary/')
from dataset import *
from DCGNet import *
from utils import *
from ply import *
import os
import json
import pandas as pd

# ====================================== Parameters ==================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--model', type=str, default='./trained_models/ae_dcg.pth', help='path to the trained model')
parser.add_argument('--classname', type=str, default='chair', help='class ids. allclass means multi-train')
parser.add_argument('--num_points', type=int, default=2500, help='number of points fed to PointNet')
parser.add_argument('--gen_points', type=int, default=2500,
                    help='30000 for dense points inference, 2500 for quantitative comparison with the baseline')
parser.add_argument('--nb_primitives1', type=int, default=5, help='number of primitives')
parser.add_argument('--nb_primitives2', type=int, default=5, help='number of primitives')
opt = parser.parse_args()
print(opt)

# ================================== Chamfer Extension ================================== #
sys.path.append("./extension/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()

# ======================================================================================== #
blue = lambda x: '\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# =================================== Create Dataset ====================================== #
dataset_test = ShapeNet(normal=False, class_choice=opt.classname, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                              shuffle=False, num_workers=int(opt.workers))

print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset_test)

# =================================== Create Network ====================================== #
network = SVR_DCGNet(nb_primitives1=opt.nb_primitives1, nb_primitives2=opt.nb_primitives2,
                     num_points=opt.num_points)
network.cuda()
network.apply(weights_init)
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("previous weight loaded")
network.eval()

# ================================== Define Stuff for Logs ================================= #
val_loss = AverageValueMeter()

# =================================== Define Grid Points =================================== #
area = opt.gen_points // (nb_primitives1 * nb_primitives2)
grain1 = int(np.sqrt(opt.gen_points / (nb_primitives1 ** 2))) - 1
grain2 = area // grain1 - 1
grain1, grain2 = grain1 * 1.0, grain2 * 1.0
vertices = []
for i in range(0, int(grain1 + 1)):
    for j in range(0, int(grain2 + 1)):
        vertices.append([i / grain1, j / grain2])
grid = [vertices]
print("grain", area, 'number vertices', len(vertices) * (opt.nb_primitives1 * opt.nb_primitives2))

# ====================================== Reset Meters ======================================= #
val_loss.reset()
for item in dataset_test.cat:
    dataset_test.perCatValueMeter[item].reset()

results = dataset_test.cat.copy()
for i in results:
    results[i] = 0

# ======================================= Testing Loop ======================================= #
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat, fn = data
        cat = cat[0]
        fn = fn[0]
        results[cat] = results[cat] + 1

        img = img.cuda()
        points = points.cuda()
        pointsReconstructed1, pointsReconstructed = network.forward_inference(img, grid)
        dist1, dist2, _, _ = distChamfer(points, pointsReconstructed)
        loss_net = torch.mean(dist1) + torch.mean(dist2)
        val_loss.update(loss_net.item())
        dataset_test.perCatValueMeter[cat].update(loss_net.item())

        outdir = './output/svr_dcg'

        if not os.path.exists(outdir + "/" + str(dataset_test.cat[cat])):
            os.makedirs(outdir + "/" + str(dataset_test.cat[cat]))
            print('created dir', outdir + "/" + str(dataset_test.cat[cat]))

        write_ply(filename=outdir + "/" + str(dataset_test.cat[cat]) + "/" + fn + "_gt",
                  points=pd.DataFrame(points.cpu().data.squeeze().numpy()), as_text=True)
        write_ply(filename=outdir + "/" + str(dataset_test.cat[cat]) + "/" + fn + "_gen" + str(
            int(opt.gen_points)), points=pd.DataFrame((pointsReconstructed.cpu().data.squeeze()).numpy()), as_text=True)
    log_table = {
        "val_loss": val_loss.avg,
        "gen_points": opt.gen_points,
    }
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    print(log_table)