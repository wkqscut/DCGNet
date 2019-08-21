"""
Defines graph encoder networks.
@GraphEncoder
"""
import torch
import torch.nn as nn
from knn import normalize_point_batch, group_knn


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_norm=True):
        super(conv2d, self).__init__()
        self.use_norm = use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        return x


class DC_Edgeconv(nn.Module):
    """Densely Connected Edgeconv"""

    def __init__(self, in_channels, growth_rate, n, k, **kwargs):
        super(DC_Edgeconv, self).__init__()
        self.growth_rate = growth_rate
        self.n = n
        self.k = k
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(conv2d(2 * in_channels, growth_rate, 1))
        for i in range(1, n):
            in_channels += growth_rate
            self.mlps.append(conv2d(in_channels, growth_rate, 1))

    def get_local_graph(self, x, k, idx=None):
        """Construct edge feature [x, NN_i - x] for each point x
        input:
            x: (B, C, N)
            k: int
            idx: (B, N, k)
        output:
            edge features: (B, C, N, k)
        """
        if idx is None:
            knn_point, idx, _ = group_knn(k + 1, x, x)
            idx = idx[:, :, 1:]
            knn_point = knn_point[:, :, :, 1:]

        neighbor_center = torch.unsqueeze(x, dim=-1)
        neighbor_center = neighbor_center.expand_as(knn_point)

        edge_feature = torch.cat(
            [neighbor_center, knn_point - neighbor_center], dim=1)
        return edge_feature, idx

    def forward(self, x, idx=None):
        for i, mlp in enumerate(self.mlps):
            if i == 0:
                y, idx = self.get_local_graph(x, k=self.k, idx=idx)
                x = x.unsqueeze(-1).repeat(1, 1, 1, self.k)
                y = torch.cat([nn.functional.relu_(mlp(y)), x], dim=1)
            elif i == (self.n - 1):
                y = torch.cat([mlp(y), y], dim=1)
            else:
                y = torch.cat([nn.functional.relu_(mlp(y)), y], dim=1)
        y, _ = torch.max(y, dim=-1)
        return y


class GraphEncoder(torch.nn.Module):
    """Graph Encoder network"""

    def __init__(self, dense_n=3, growth_rate=12, knn=16):
        super(GraphEncoder, self).__init__()
        self.dense_n = dense_n
        self.layer0 = conv2d(3, 24, [1, 1])
        self.layer1 = DC_Edgeconv(
            24, growth_rate=growth_rate, n=dense_n, k=knn)

    def forward(self, xyz):
        batch_size, _, num_point = xyz.size()
        # normalize
        xyz_normalized, centroid, radius = normalize_point_batch(
            xyz, NCHW=True)
        x = self.layer0(xyz_normalized.unsqueeze(dim=-1)).squeeze(dim=-1)
        point_features = torch.cat([self.layer1(x), x], dim=1)
        return point_features


if __name__ == '__main__':
    points = torch.rand(32, 3, 500)
    model = GraphEncoder().cuda()
    print(model)
    point_features = model(points.cuda())
    print(point_features.size())
