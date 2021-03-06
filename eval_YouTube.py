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


torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    # parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("--vos_root", type=str, help="path to data", default='../YouTube/vos/train')
    parser.add_argument("--af_root", type=str, help="path to data", default='../YouTube/vos/all_frames/train')

    parser.add_argument("--id", type=int, help='Id out of total ID for partitioning', default=0)
    parser.add_argument("--start", type=int, help='Start IDX  (inclusive)', default=0)
    parser.add_argument("--end", type=int, help='END IDX (inclusive)', default=0)

    parser.add_argument("--extra_id", type=str, default='')

    parser.add_argument("--before", type=int, help='Memory before')
    parser.add_argument("--after", type=int, help='Memory after')

    return parser.parse_args()

args = get_arguments()

# GPU = args.g
VIZ = args.viz
VOS_ROOT = args.vos_root
AF_ROOT = args.af_root

id = args.id
start_idx = args.start
end_idx = args.end
before = args.before
after = args.after

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on YouTube')

# os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

palette = Image.open(path.join(VOS_ROOT, 'Annotations/0a2f2bd294/00000.png')).getpalette()

def Run_video(Fs, Ms, AFs, mem_before, mem_after, num_objects):

    # print(Fs.shape, Ms.shape, AFs.shape)
    b, _, t, w, h = Fs.shape
    _, _, at, w, h = AFs.shape
    _, k, _, _, _ = Ms.shape

    all_keys = [None] * t
    all_values = [None] * t
    for i in range(t):
        all_keys[i], all_values[i] = model(Fs[:,:,i], Ms[:,:,i], torch.tensor([num_objects]))
        all_keys[i] = all_keys[i].cpu()
        all_values[i] = all_values[i].cpu()

    all_keys = torch.cat(all_keys, 3)
    all_values = torch.cat(all_values, 3)

    Es = torch.zeros((b, k, at, w, h), dtype=torch.float32, device=Ms.device)
    Es[:,:,0] = Ms[:,:,0]

    # for t_step in tqdm.tqdm(range(1, num_frames)):
    for t_step in range(1, at):
        # memorize
        prev_key, prev_value = model(AFs[:,:,t_step-1], Es[:,:,t_step-1], torch.tensor([num_objects]))
        prev_key = prev_key.cpu()
        prev_value = prev_value.cpu()

        # if t-1 == 0: # 
        #     this_keys, this_values = prev_key, prev_value # only prev memory
        # else:
        #     this_keys = torch.cat([keys, prev_key], dim=3)
        #     this_values = torch.cat([values, prev_value], dim=3)

        inter_idx = max(0, min(t-1, (t_step+4)//5))
        start_idx = max(0, inter_idx - mem_before)
        end_idx = min(t-1, inter_idx + mem_after + 1)

        be_keys = all_keys[:,:,:, start_idx : inter_idx]
        be_values = all_values[:,:,:, start_idx : inter_idx]

        af_keys = all_keys[:,:,:, inter_idx : end_idx]
        af_values = all_values[:,:,:, inter_idx : end_idx]

        # print(start_idx, inter_idx, end_idx)
        # print(be_keys.shape, af_keys.shape, prev_key.shape)

        this_keys = torch.cat([be_keys, af_keys, prev_key], 3).cuda()
        this_values = torch.cat([be_values, af_values, prev_value], 3).cuda()

        # this_keys = torch.cat([all_keys, prev_key], dim=3)
        # this_values = torch.cat([all_values, prev_value], dim=3)
        
        # segment
        logit = model(AFs[:,:,t_step], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t_step] = F.softmax(logit, dim=1)
        
        # # update
        # if t-1 in to_memorize:
        #     keys, values = this_keys, this_values
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es



Testset = YOUTUBE_VOS_MO_Test(VOS_ROOT, AF_ROOT, start_idx=start_idx, end_idx=end_idx)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = 'YouTube_%d_%d' % (start_idx, end_idx)
skipped = []

for seq, V in progressbar(enumerate(Testloader), max_value=len(Testloader), redirect_stdout=True):
    Fs, Ms, AFs, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    num_objects = info['num_objects'][0]
    frames_name = info['frames_name']

    print(seq_name)
    
    with torch.no_grad():
        try:
            pred, Es = Run_video(Fs, Ms, AFs, mem_before=before, mem_after=after, num_objects=num_objects)
        except RuntimeError as e:
            print('Exception', e, seq_name)
            skipped.append(seq_name)
            print('Skipped: ', skipped)
        
    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

