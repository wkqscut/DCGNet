"""
DCGNet for Point Set Reconstruction from a Single Image.
@train_SVR_DCGNet.py
"""

from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from dataset import *
from DCGNet import *
from utils import *
from ply import *
import os
import json
import time, datetime
import visdom

best_val_loss = 10

# ====================================== Parameters ==================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--model_preTrained_AE', type=str, default='trained_models/ae_dcg.pth', help='model path')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='number of points')
parser.add_argument('--classname', type=str, default='chair', help='class ids. None means multi-train')  # True
parser.add_argument('--nb_primitives1', type=int, default=5, help='number of primitives1')
parser.add_argument('--nb_primitives2', type=int, default=5, help='number of primitives2')
parser.add_argument('--env', type=str, default="SVR_AtlasNet", help='visdom env')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay1', type=int, default=300, help='lr_decay_epoch1')
parser.add_argument('--lr_decay2', type=int, default=400, help='lr_decay_epoch2')
parser.add_argument('--fix_decoder', type=bool, default=False,
                    help='if set to True, only train the resnet encoder')

opt = parser.parse_args()
print(opt)

# ================================== Chamfer Extension ================================== #
sys.path.append("./extension/")
import dist_chamfer as ext

distChamfer = ext.chamferDist()

# ======================================================================================== #
# Launch visdom for visualization

vis = visdom.Visdom(port=8990, env=opt.env)
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('log', 'svr_dcg', opt.classname, save_path)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
logname = os.path.join(dir_name, 'log.txt')

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# =================================== Create Dataset ====================================== #
dataset = ShapeNet(SVR=True, normal=False, class_choice=opt.classname, train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataset_test = ShapeNet(SVR=True, normal=False, class_choice=opt.classname, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

dataset_test_view = ShapeNet(SVR=True, normal=False, class_choice=opt.classname, train=True, gen_view=True)
dataloader_test_view = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                                   shuffle=False, num_workers=int(opt.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset)

# =================================== Load PreTrained AE Network ========================== #
if model_preTrained_AE != '':
    network_preTrained_autoencoder = AE_DCGNet_DGCNN_PointNet(num_points=opt.num_points,
                                                              nb_primitives=opt.nb_primitives1)
    network_preTrained_autoencoder.cuda()
    network_preTrained_autoencoder.load_state_dict(torch.load(opt.model_preTrained_AE))
    val_loss = AverageValueMeter()
    val_loss.reset()
    network_preTrained_autoencoder.eval()
    for i, data in enumerate(dataloader_test, 0):
        img, points, cat, _, _ = data
        points = points.cuda()
        _, pointsReconstructed = network_preTrained_autoencoder(points)
        dist1, dist2, _, _ = distChamfer(points, pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        val_loss.update(loss_net.item())
    print("Previous decoder performances : ", val_loss.avg)

# =================================== Create Network ====================================== #
network = SVR_DCGNet(nb_primitives1=opt.nb_primitives1, nb_primitives2=opt.nb_primitives2, num_points=opt.num_points)
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

if opt.fix_decoder:
    network.decoder1 = network_preTrained_autoencoder.decoder1
    network.decoder2 = network_preTrained_autoencoder.decoder2
    network.dgcnn = network_preTrained_autoencoder.dgcnn
    network_preTrained_autoencoder.cpu()
    network.cuda()
print(network)

# ==================================== Create Optimizer ===================================== #
lrate = opt.lr
params_dict = dict(network.named_parameters())
params = []

if opt.fix_decoder:
    optimizer = optim.Adam(network.encoder.parameters(), lr=lrate)
else:
    optimizer = optim.Adam(network.parameters(), lr=lrate)

# =================================== DEFINE stuff for logs ================================== #
num_batch = len(dataset) / opt.batchSize

train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
val_view_loss = AverageValueMeter()
with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')

trainloss_acc0 = 1e-9
trainloss_accs = 0

train_curve = []
val_curve = []
val_view_curve = []
labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives1 * opt.nb_primitives2 + 1) * (
        opt.num_points // (opt.nb_primitives1 * opt.nb_primitives2)) + 1)).view(
    opt.num_points // (opt.nb_primitives1 * opt.nb_primitives2),
    ((opt.nb_primitives1 * opt.nb_primitives2) + 1)).transpose(0, 1)
labels_generated_points = (labels_generated_points) % ((opt.nb_primitives1 * opt.nb_primitives2) + 1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
print(labels_generated_points)

# ===================================== Training Loop ========================================= #
for epoch in range(opt.nepoch):
    # TRAIN MODE
    if epoch == opt.lr_decay1:
        optimizer = optim.Adam(network.parameters(), lr=opt.lr / 10.0)
        lrate = lrate / 10.0
    if epoch == opt.lr_decay2:
        optimizer = optim.Adam(network.parameters(), lr=opt.lr / 100.0)
        lrate = lrate / 10.0
    print('lr is set to: ', lrate)

    train_loss.reset()
    network.train()

    if epoch <= opt.lr_decay2:
        alpha = exponential_decay(epoch, init=0.99, m=opt.lr_decay2, finish=0.01)
        print('alpha is set to :', alpha)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, cat, _ = data
        img = img.cuda()
        points = points.cuda()
        pointsReconstructed1, pointsReconstructed = network.forward(img)
        dist11, dist12, _, _ = distChamfer(points, pointsReconstructed1)
        dist1, dist2, _, _ = distChamfer(points, pointsReconstructed)
        loss_net1 = (torch.mean(dist11)) + (torch.mean(dist12))
        loss_net2 = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net = alpha * loss_net1 + (1.00 - alpha) * loss_net2
        loss_net.backward()
        train_loss.update(loss_net.item())

        optimizer.step()

        # VIZUALIZE
        if i % 50 <= 0:
            vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN"))
            vis.scatter(X=points[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title="TRAIN_INPUT",
                            markersize=2,
                        ),
                        )
            vis.scatter(X=pointsReconstructed[0].data.cpu(),
                        Y=labels_generated_points[0:pointsReconstructed.size(1)],
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )
        print('[%d: %d/%d] train loss:  %f' % (
            epoch, i, len_dataset / opt.batchSize, loss_net.item()))

    # UPDATE CURVES
    train_curve.append(train_loss.avg)

    with torch.no_grad():

        # VALIDATION on same models new views
        if epoch % 10 == 0:
            val_view_loss.reset()
            network.eval()
            for i, data in enumerate(dataloader_test_view, 0):
                img, points, cat, _ = data
                img = img.cuda()
                points = points.cuda()
                _, pointsReconstructed = network(img)
                dist1, dist2, _, _ = distChamfer(points, pointsReconstructed)
                loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
                val_view_loss.update(loss_net.item())
            # UPDATE CURVES
        val_view_curve.append(val_view_loss.avg)

        # VALIDATION
        val_loss.reset()
        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()

        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            img, points, cat, _ = data
            img = img.cuda()
            points = points.cuda()
            _, pointsReconstructed = network(img)
            dist1, dist2, _, _ = distChamfer(points, pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            val_loss.update(loss_net.item())
            dataset_test.perCatValueMeter[cat[0]].update(loss_net.item())
            if i % 25 == 0:
                vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE TRAIN"))
                vis.scatter(X=points[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsReconstructed[0].data.cpu(),
                            Y=labels_generated_points[0:pointsReconstructed.size(1)],
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )
            print('[%d: %d/%d] val loss:  %f ' % (epoch, i, len(dataset_test), loss_net.item()))

        # UPDATE CURVES
        val_curve.append(val_loss.avg)

    vis.line(
        X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)), np.arange(len(val_view_curve)))),
        Y=np.column_stack((np.array(train_curve), np.array(val_curve), np.array(val_view_curve))),
        win='loss',
        opts=dict(title="loss", legend=["train_curve" + opt.env, "val_curve" + opt.env, "val_view_curve" + opt.env],
                  markersize=2, ), )
    vis.line(
        X=np.column_stack((np.arange(len(train_curve)), np.arange(len(val_curve)), np.arange(len(val_view_curve)))),
        Y=np.log(np.column_stack((np.array(train_curve), np.array(val_curve), np.array(val_view_curve)))),
        win='log',
        opts=dict(title="log", legend=["train_curve" + opt.env, "val_curve" + opt.env, "val_view_curve" + opt.env],
                  markersize=2, ), )

    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "val_loss_new_views_same_models": val_view_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "alpha": alpha,
        "bestval": best_val_loss,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    # save last network
    if best_val_loss > val_loss.avg:
        best_val_loss = val_loss.avg
        print('New best loss : ', best_val_loss)
        print('saving net...')
        torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
