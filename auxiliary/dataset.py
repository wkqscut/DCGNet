"""
Defines object dataset.
@ShapeNet
"""
from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import *


class ShapeNet(data.Dataset):
    def __init__(self, rootimg="./data/ShapeNetRendering", rootpc="./data/ShapeNetPoints", class_choice="chair",
                 train=True, npoints=2500, normal=False, gen_view=False, SVR=False, idx=0):
        self.normal = normal
        self.train = train
        self.rootimg = rootimg
        self.rootpc = rootpc
        self.npoints = npoints
        self.datapath = []
        self.catfile = os.path.join('./data/synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        self.SVR = SVR
        self.gen_view = gen_view
        self.idx = idx
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice == 'allclass':
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            dir_img = os.path.join(self.rootimg, self.cat[item])
            fns_img = sorted(os.listdir(dir_img))

            try:
                dir_point = os.path.join(self.rootpc, self.cat[item])
                fns_pc = sorted(os.listdir(dir_point))
            except:
                fns_pc = []
            fns = [val for val in fns_img if val + '.points.ply' in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns) / float(len(fns_img)), "%"),
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]

            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    self.meta[item].append((os.path.join(dir_img, fn, "rendering"),
                                            os.path.join(dir_point, fn + '.points.ply'), item, fn))
            else:
                empty.append(item)
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
            # normalize,
        ])

        # RandomResizedCrop or RandomCrop
        self.dataAugmentation = transforms.Compose([
            transforms.RandomCrop(127),
            transforms.RandomHorizontalFlip(),
        ])
        self.validating = transforms.Compose([
            transforms.CenterCrop(127),
        ])

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()
        self.perCatValueMeter_metro = {}
        for item in self.cat:
            self.perCatValueMeter_metro[item] = AverageValueMeter()
        self.transformsb = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
        ])

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn[1]) as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    try:
                        lenght = int(line.split()[2])
                    except ValueError:
                        print(fn)
                        print(line)
                    break
        for i in range(15):
            try:
                mystring = my_get_n_random_lines(fn[1], n=self.npoints)
                mypoint_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)

        if not self.normal:
            point_set = mypoint_set[:,0:3]
        else:
            point_set = mypoint_set[:,0:3]
            normal = 0.1 * mypoint_set[:,3:6]
            normal = torch.from_numpy(normal)
        point_set = torch.from_numpy(point_set)

        # load image
        if self.SVR:
            if self.train:
                N_tot = len(os.listdir(fn[0])) - 3
                if N_tot == 1:
                    print("only one view in ", fn)
                if self.gen_view:
                    N = 0
                else:
                    N = np.random.randint(1, N_tot)
                if N < 10:
                    im = Image.open(os.path.join(fn[0], "0" + str(N) + ".png"))
                else:
                    im = Image.open(os.path.join(fn[0], str(N) + ".png"))

                im = self.dataAugmentation(im)  # random crop
            else:
                if self.idx < 10:
                    im = Image.open(os.path.join(fn[0], "0" + str(self.idx) + ".png"))
                else:
                    im = Image.open(os.path.join(fn[0], str(self.idx) + ".png"))
                im = self.validating(im)  # center crop
            data = self.transforms(im)  # scale
            data = data[:3, :, :]
        else:
            data = 0
        if not self.normal:
            return data, point_set.contiguous(), fn[2], fn[3]
        else:
            return data, point_set.contiguous(), normal.contiguous(), fn[2], fn[3]
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':

    print('Testing Shapenet dataset')
    d = ShapeNet(class_choice='allclass', normal=False, train=True, npoints=2500)
    a = len(d)
    d = ShapeNet(class_choice='allclass', normal=False, train=False, npoints=2500)
    a = a + len(d)
    print(a)
    dataloader_test = torch.utils.data.DataLoader(d, batch_size=32,
                                                  shuffle=False, num_workers=4)
    for i, data in enumerate(dataloader_test):
        img, point, _, _ = data
        print(point, point.shape)
        break
