from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
from os import path
import argparse
import copy

from progressbar import progressbar

### My libs
from youtube_dataset import YOUTUBE_VOS_MO_Test
from model import STM

from bb_utils import *


torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    # parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("--vos_root", type=str, help="path to data", default='../YouTube/vos/train_480p')
    parser.add_argument("--af_root", type=str, help="path to data", default='../YouTube/vos/all_frames/train_480p')

    parser.add_argument("--id", type=int, help='Id out of total ID for partitioning', default=0)
    parser.add_argument("--start", type=int, help='Start IDX  (inclusive)', default=0)
    parser.add_argument("--end", type=int, help='END IDX (inclusive)', default=0)

    return parser.parse_args()

args = get_arguments()

# GPU = args.g
VIZ = args.viz
VOS_ROOT = args.vos_root
AF_ROOT = args.af_root

id = args.id
start_idx = args.start
end_idx = args.end

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on YouTube')

# os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')


palette = Image.open(path.join(VOS_ROOT, 'Annotations/0a2f2bd294/00000.png')).getpalette()

def Run_video(Fs, Ms, AFs, num_objects, info):

    skip_frames_names = info['skip_frames_name']

    # print(Fs.shape, Ms.shape, AFs.shape)
    b, _, t, h, w = Fs.shape
    _, _, at, h, w = AFs.shape
    _, k, _, _, _ = Ms.shape

    for ref_idx in range(t):

        ref_key, ref_value = model(Fs[:,:,ref_idx], Ms[:,:,ref_idx], num_objects=torch.tensor([num_objects]))

        Es = torch.zeros((b, k, at, h, w), dtype=torch.float32, device=Ms.device)
        Es[:,:,0] = Ms[:,:,0]

        # for t_step in tqdm.tqdm(range(1, num_frames)):
        for t_step in range(1, at):
            # segment
            logit = model(AFs[:,:,t_step], ref_key, ref_value, torch.tensor([num_objects]))
            Es[:,:,t_step] = F.softmax(logit, dim=1)
        
        pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)

        test_path = os.path.join('./test', code_name, seq_name, skip_frames_names[ref_idx][0][:-4])
        os.makedirs(test_path, exist_ok=True)
        for i, f_name in enumerate(frames_name):
            img_E = Image.fromarray(pred[i])
            img_E.putpalette(palette)
            img_E.save(os.path.join(test_path, f_name[0].replace('.jpg', '.png')))


Testset = YOUTUBE_VOS_MO_Test(VOS_ROOT, AF_ROOT, start_idx, end_idx)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = 'YouTube_one_ref_%d_%d' % (id, total_id)
code_name += args.extra_id

skipped = []

for seq, V in progressbar(enumerate(Testloader), max_value=len(Testloader), redirect_stdout=True):
    Fs, Ms, AFs, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    num_objects = info['num_objects'][0]
    frames_name = info['frames_name']

    Fs = Fs.cuda(non_blocking=True)
    AFs = AFs.cuda(non_blocking=True)
    Ms = Ms.cuda(non_blocking=True)
    
    print(seq_name)

    with torch.no_grad():
        try:
            Run_video(Fs, Ms, AFs, num_objects=num_objects, info=info)
        except RuntimeError as e:
            print('Exception', e, seq_name)
            skipped.append(seq_name)
            print('Skipped: ', skipped)
            # Run_video(Fs, Ms, AFs, num_objects=num_objects, info=info)

    del Fs
    del AFs
    del Ms
        
    # Save results for quantitative eval ######################

print('Total skipped: ', skipped)

