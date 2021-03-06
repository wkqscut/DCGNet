"""
Demo of DCGNet.
@run_SVR_DCGNet.py
"""

from __future__ import print_function
import argparse
import random
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('./auxiliary/')
from dataset import *
from DCGNet import *
from utils import *
from ply import *
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="data/plane_input_demo.png", help='input image')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--model', type=str, default='', help='path to the trained model')
parser.add_argument('--classname', type=str, default='chair', help='class ids. allclass means multi-train')
parser.add_argument('--num_points', type=int, default=2500, help='number of points fed to PointNet')
parser.add_argument('--gen_points', type=int, default=30000,
                    help='30000 for dense points inference, 2500 for sparse points inference')
parser.add_argument('--nb_primitives1', type=int, default=5, help='number of primitives')
parser.add_argument('--nb_primitives2', type=int, default=5, help='number of primitives')
parser.add_argument('--cuda', type=int, default=1, help='use cuda')

opt = parser.parse_args()

blue = lambda x: '\033[94m' + x + '\033[0m'
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

network = SVR_DCGNet(nb_primitives1=opt.nb_primitives1, nb_primitives2=opt.nb_primitives2,
                     num_points=opt.num_points)
if opt.cuda:
    network.cuda()

network.apply(weights_init)

if opt.cuda:
    network.load_state_dict(torch.load(opt.model))
else:
    network.load_state_dict(torch.load(opt.model, map_location='cpu'))
print("previous weight loaded")

network.eval()

grain = int(np.sqrt(opt.gen_points/(opt.nb_primitives1 * opt.nb_primitives2)))-1
grain = grain*1.0
print(grain)

vertices = []
for i in range(0, int(grain + 1)):
    for j in range(0, int(grain + 1)):
        vertices.append([i / grain, j / grain])
grid = [vertices for i in range(0,opt.nb_primitives1)]
print("grain", grain, 'number vertices', len(vertices) * (opt.nb_primitives1 * opt.nb_primitives2))

# prepare the input data
my_transforms = transforms.Compose([
    transforms.CenterCrop(127),
    transforms.Resize(size=224, interpolation=2),
    transforms.ToTensor(),
    # normalize,
])

im = Image.open(opt.input)
im = my_transforms(im)  # scale
img = im[:3, :, :].unsqueeze(0)

img = Variable(img)
if opt.cuda:
    img = img.cuda()

# forward pass
pointsReconstructed1, pointsReconstructed = network.forward_inference(img, grid)

outdir = './data/svr_demo'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    print('created dir', outdir)

# Save output points
write_ply(filename=outdir +  "/" +  "demo_gen" + str(
    int(opt.gen_points)), points=pd.DataFrame((pointsReconstructed.cpu().data.squeeze()).numpy()), as_text=True)

print("Done demoing! Check out results in data/")
