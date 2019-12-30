"""
Defines networks.
@SVR_DCG_BackboneNet
@SVR_DCGNet
@AE_PointNet_DCG_BackboneNet
@AE_PointNet_DCGNet
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import resnet
from graph_layers import *


# ========================================= Point Set Encoder ================================== #
class STN3d(nn.Module):
    """Spatial Transformer Network"""

    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    """PointNet for Feature Extraction"""

    def __init__(self, num_points=2500, global_feat=True, trans=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


# ======================================= Point Set Decoder ==================================== #
class ResPointGenCon(nn.Module):
    """Multilayer Perceptrons"""

    def __init__(self, bottleneck_size=2500, residual=True):
        self.bottleneck_size = bottleneck_size
        super(ResPointGenCon, self).__init__()
        self.residual = residual
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x, initial_points):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))

        if self.residual:
            x = x + initial_points
        return x


# =========================================== DCGNet =========================================== #
class SVR_DCG_BackboneNet(nn.Module):
    """DCG_BackboneNet for Single View Reconstruction"""

    def __init__(self, bottleneck_size=1024, num_points=2500, nb_primitives1=5, nb_primitives2=5,
                 pretrained_encoder=False):
        super(SVR_DCG_BackboneNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives1 = nb_primitives1
        self.nb_primitives2 = nb_primitives2
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder1 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=2 + self.bottleneck_size) for _ in range(0, self.nb_primitives1)])
        self.decoder2 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=3 + self.bottleneck_size) for _ in range(0, self.nb_primitives2)])

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)

        outs = []
        # ================================ Stage One ================================ #
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // (
                    self.nb_primitives1 * self.nb_primitives2)))  # [bs, 2, 100]
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()  # [bs, 1024, 100]
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()  # [bs, 1026, 100]
            outs.append(self.decoder1[i](y, 0))  # [bs, 3, 100]
        coarse_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3 500]

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()  # [bs, 1024, 500]
            y = torch.cat((coarse_point_sets, y), 1).contiguous()  # [bs, 1024+3, 500]
            outs.append(self.decoder2[i](y, coarse_point_sets))  # [bs, 3, 500]
        fine_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3, 2500]

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            output = self.decoder1[i](y, 0)
            outs.append(output)
        coarse_point_sets = torch.cat(outs, 2).contiguous()

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()
            y = torch.cat((coarse_point_sets, y), 1).contiguous()
            outs.append(self.decoder2[i](y, coarse_point_sets))
        fine_point_sets = torch.cat(outs, 2).contiguous()
        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_latent(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        return x


class SVR_DCGNet(nn.Module):
    """DCGNet for Single View Reconstruction"""

    def __init__(self, bottleneck_size=1024, num_points=2500, nb_primitives1=5, nb_primitives2=5,
                 pretrained_encoder=False):
        super(SVR_DCGNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives1 = nb_primitives1
        self.nb_primitives2 = nb_primitives2
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder1 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=2 + self.bottleneck_size) for _ in range(0, self.nb_primitives1)])
        self.decoder2 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=3 + 84 + self.bottleneck_size) for _ in range(0, self.nb_primitives2)])
        self.dgcnn = GraphEncoder(dense_n=3, growth_rate=12, knn=16)  # 84 96 108

    def forward(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // (
                    self.nb_primitives1 * self.nb_primitives2)))  # [bs, 2, 100]
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()  # [bs, 1024, 100]
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()  # [bs, 1026, 100]
            outs.append(self.decoder1[i](y, 0))  # [bs, 3, 100]
        coarse_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3 500]
        graphfeat = self.dgcnn(coarse_point_sets)

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()  # [bs, 1024, 500]
            y = torch.cat((coarse_point_sets, graphfeat, y), 1).contiguous()  # [bs, 1024+3+84, 500]
            outs.append(self.decoder2[i](y, coarse_point_sets))  # [bs, 3, 500]
        fine_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3, 2500]

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder1[i](y, 0))
        coarse_point_sets = torch.cat(outs, 2).contiguous()
        graphfeat = self.dgcnn(coarse_point_sets)

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()
            y = torch.cat((coarse_point_sets, graphfeat, y), 1).contiguous()
            output = self.decoder2[i](y, coarse_point_sets)
            outs.append(output)
        fine_point_sets = torch.cat(outs, 2).contiguous()

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_latent(self, x):
        x = x[:, :3, :, :].contiguous()
        x = self.encoder(x)
        return x


class AE_PointNet_DCG_BackboneNet(nn.Module):
    """DCG_BackboneNet for Shape AutoEncoding"""

    def __init__(self, bottleneck_size=1024, num_points=2500, nb_primitives1=5, nb_primitives2=5,
                 pretrained_encoder=False):
        super(AE_PointNet_DCG_BackboneNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives1 = nb_primitives1
        self.nb_primitives2 = nb_primitives2
        self.pretrained_encoder = pretrained_encoder
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU())
        self.decoder1 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives1)])
        self.decoder2 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=3 + 84 + self.bottleneck_size) for i in range(0, self.nb_primitives2)])
        self.dgcnn = GraphEncoder(dense_n=3, growth_rate=12, knn=16)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // (
                    self.nb_primitives1 * self.nb_primitives2)))  # [bs, 2, 100]
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()  # [bs, 1024, 100]
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()  # [bs, 1026, 100]
            outs.append(self.decoder1[i](y, 0))  # [bs, 3, 100]
        coarse_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3 500]
        graphfeat = self.dgcnn(coarse_point_sets)

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()  # [bs, 1024, 500]
            y = torch.cat((coarse_point_sets, graphfeat, y), 1).contiguous()  # [bs, 1024+3+84, 500]
            outs.append(self.decoder2[i](y, coarse_point_sets))  # [bs, 3, 500]
        fine_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3, 2500]

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = x.transpose(2, 1).contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder1[i](y, 0))
        coarse_point_sets = torch.cat(outs, 2).contiguous()
        graphfeat = self.dgcnn(coarse_point_sets)

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()
            y = torch.cat((coarse_point_sets, graphfeat, y), 1).contiguous()
            output = self.decoder2[i](y, coarse_point_sets)
            outs.append(output)
        fine_point_sets = torch.cat(outs, 2).contiguous()

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_latent(self, x):
        x = x.transpose(2, 1).contiguous()
        x = self.encoder(x)
        return x


class AE_PointNet_DCGNet(nn.Module):
    """DCGNet for Shape AutoEncoding"""

    def __init__(self, bottleneck_size=1024, num_points=2500, nb_primitives1=5, nb_primitives2=5,
                 pretrained_encoder=False):
        super(AE_PointNet_DCGNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives1 = nb_primitives1
        self.nb_primitives2 = nb_primitives2
        self.pretrained_encoder = pretrained_encoder
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU())
        self.decoder1 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives1)])
        self.decoder2 = nn.ModuleList(
            [ResPointGenCon(bottleneck_size=3 + 84 + self.bottleneck_size) for i in range(0, self.nb_primitives2)])
        self.dgcnn = GraphEncoder(dense_n=3, growth_rate=12, knn=16)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // (
                    self.nb_primitives1 * self.nb_primitives2)))  # [bs, 2, 100]
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()  # [bs, 1024, 100]
            y = torch.cat((rand_grid, y.type_as(rand_grid)), 1).contiguous()  # [bs, 1026, 100]
            outs.append(self.decoder1[i](y, 0))  # [bs, 3, 100]
        coarse_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3 500]
        graphfeat = self.dgcnn(coarse_point_sets)

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()  # [bs, 1024, 500]
            y = torch.cat((coarse_point_sets, graphfeat, y), 1).contiguous()  # [bs, 1024+3+84, 500]
            outs.append(self.decoder2[i](y, coarse_point_sets))  # [bs, 3, 500]
        fine_point_sets = torch.cat(outs, 2).contiguous()  # [bs, 3, 2500]

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_inference(self, x, grid):
        x = x.transpose(2, 1).contiguous()
        x = self.encoder(x)

        # ================================ Stage One ================================ #
        outs = []
        for i in range(0, self.nb_primitives1):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder1[i](y, 0))
        coarse_point_sets = torch.cat(outs, 2).contiguous()
        graphfeat = self.dgcnn(coarse_point_sets)

        # ================================ Stage Two ================================ #
        outs = []
        for i in range(0, self.nb_primitives2):
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), coarse_point_sets.size(2)).contiguous()
            y = torch.cat((coarse_point_sets, graphfeat, y), 1).contiguous()
            output = self.decoder2[i](y, coarse_point_sets)
            outs.append(output)
        fine_point_sets = torch.cat(outs, 2).contiguous()

        return coarse_point_sets.transpose(2, 1).contiguous(), fine_point_sets.transpose(2, 1).contiguous()

    def forward_latent(self, x):
        x = x.transpose(2, 1).contiguous()
        x = self.encoder(x)
        return x


if __name__ == '__main__':
    grain = 49.0
    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain])
    grid = [vertices]
    #print(grid)

    print('testing DCGNet...')
    img = Variable(torch.rand(32, 3, 224, 224))
    # points = Variable(torch.rand(32, 2500, 3))
    model = SVR_DCGNet().cuda()
    model.cuda()
    print(model)
    coarse_pts, fine_pts = model(img.cuda())
    print('output points: ', fine_pts.size())
