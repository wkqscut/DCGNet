"""
Defines Hausdorff Distance.
@HausdorffDist
"""
import math
import numpy as np
from numpy.core.umath_tests import inner1d
import torch


def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2 * zz)
    return P


# =============================== Hausdorff Distance ===============================#
def HausdorffDist(A, B):
    # Find pairwise distance
    # D_mat = torch.sqrt((torch.sum(A*A,dim=1)).view(-1,1)+ torch.sum(B*B,dim=1)-2*(torch.mm(A,B.t())))
    D_mat = torch.sqrt(pairwise_dist(A, B))
    # Find DH
    dH = torch.max(torch.max(torch.min(D_mat, dim=0)[0]), torch.max(torch.min(D_mat, dim=1)[0]))
    return dH
