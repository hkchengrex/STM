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
    parser.add_argument("--vos_root", type=str, help="path to data", default='../YouTube/vos/train')
    parser.add_argument("--af_root", type=str, help="path to data", default='../YouTube/vos/all_frames/train')

    parser.add_argument("--id", type=int, help='Id out of total ID for partitioning', default=0)
    parser.add_argument("--total_id", type=int, help='Total ID for partitioning', default=1)
    parser.add_argument("--start_idx", type=int, help='Skip some index in the current partition', default=0)

    parser.add_argument("--before", type=int, help='Memory before', default=3)
    parser.add_argument("--after", type=int, help='Memory after', default=3)

    parser.add_argument('--extra_id', default='')

    return parser.parse_args()

args = get_arguments()

# GPU = args.g
VIZ = args.viz
VOS_ROOT = args.vos_root
AF_ROOT = args.af_root

id = args.id
total_id = args.total_id
start_idx = args.start_idx
before = args.before
after = args.after

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

def Run_video(Fs, Ms, AFs, mem_before, mem_after, num_objects):

    # print(Fs.shape, Ms.shape, AFs.shape)
    b, _, t, h, w = Fs.shape
    _, _, at, h, w = AFs.shape
    _, k, _, _, _ = Ms.shape

    all_keys = [None] * t
    all_values = [None] * t
    for i in range(t):
        all_keys[i], all_values[i] = model(Fs[:,:,i], Ms[:,:,i], num_objects=torch.tensor([num_objects]))
        all_keys[i] = all_keys[i].cpu()
        all_values[i] = all_values[i].cpu()

    all_keys = torch.cat(all_keys, 3)
    all_values = torch.cat(all_values, 3)

    Es = torch.zeros((b, k, at, h, w), dtype=torch.float32, device=Ms.device)
    Es[:,:,0] = Ms[:,:,0]

    # for t_step in tqdm.tqdm(range(1, num_frames)):
    for t_step in range(1, at):
        # memorize
        prev_key, prev_value = model(AFs[:,:,t_step-1], Es[:,:,t_step-1], num_objects=torch.tensor([num_objects]))
        class_prob = torch.zeros((k-1, h, w), dtype=torch.float32)

        for obj_idx in range(1, num_objects+1):

            inter_idx = max(0, min(t-1, t_step//5))
            start_idx = max(0, inter_idx - mem_before)
            end_idx = min(t, inter_idx + mem_after + 1)

            # print(t_step, inter_idx, start_idx, end_idx)

            last_mask = Ms[:,obj_idx,inter_idx]
            if inter_idx != t-1:
                next_mask = Ms[:,obj_idx,inter_idx+1]
                combined_mask = last_mask + next_mask
                scale = 0.1
            else:
                # No future GT frames, just go with this
                combined_mask = last_mask
                scale = 0.25
            try:
                coords = get_bb_position(combined_mask.numpy()[0])
                coords = scale_bb_by(*coords, h, w, scale, scale)
                # print('pre', *coords)
                rmin, rmax, cmin, cmax = fit_bb_to_stride(*coords, h, w, 16)
            except:
                # print('Cannot find BB')
                # Cannot find BB --> Nothing to see here!
                continue

            # print(all_keys.shape, all_values.shape)
            # print('afr', rmin, rmax, cmin, cmax)

            be_keys = all_keys[:,obj_idx:obj_idx+1, :, start_idx:inter_idx+1, rmin//16:rmax//16+1, cmin//16:cmax//16+1]
            be_values = all_values[:,obj_idx:obj_idx+1, :, start_idx:inter_idx+1, rmin//16:rmax//16+1, cmin//16:cmax//16+1]

            af_keys = all_keys[:,obj_idx:obj_idx+1, :, inter_idx+1:end_idx, rmin//16:rmax//16+1, cmin//16:cmax//16+1]
            af_values = all_values[:,obj_idx:obj_idx+1, :, inter_idx+1:end_idx, rmin//16:rmax//16+1, cmin//16:cmax//16+1]

            this_keys = torch.cat([be_keys, af_keys], 3).cuda()
            this_values = torch.cat([be_values, af_values], 3).cuda()

            prev_key_obj = prev_key[:, obj_idx:obj_idx+1,:,:, rmin//16:rmax//16+1, cmin//16:cmax//16+1]
            prev_value_obj = prev_value[:, obj_idx:obj_idx+1,:,:, rmin//16:rmax//16+1, cmin//16:cmax//16+1]

            this_keys = torch.cat([this_keys, prev_key_obj], 3)
            this_values = torch.cat([this_values, prev_value_obj], 3)

            # segment
            ps = model(AFs[:,:,t_step,rmin:rmax+1,cmin:cmax+1], this_keys, this_values, num_objects=torch.tensor([1]), only_obj=True)
            class_prob[obj_idx-1, rmin:rmax+1, cmin:cmax+1] = ps
            
        logit = model.module.Soft_aggregation(class_prob, k)
        Es[:,:,t_step] = F.softmax(logit, dim=1).cpu()
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es



Testset = YOUTUBE_VOS_MO_Test(VOS_ROOT, AF_ROOT, id=id, total_id=total_id, skip_idx=start_idx)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = 'YouTube_BB_b%d_a%d' % (before, after)
code_name += args.extra_id

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
            torch.cuda.empty_cache()
            pred, Es = Run_video(Fs, Ms, AFs, mem_before=before, mem_after=after, num_objects=num_objects)
        
    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for i, f_name in enumerate(frames_name):
        img_E = Image.fromarray(pred[i])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, f_name[0].replace('.jpg', '.png')))

    # if VIZ:
    #     from helpers import overlay_davis
    #     # visualize results #######################
    #     viz_path = os.path.join('./viz/', code_name, seq_name)
    #     if not os.path.exists(viz_path):
    #         os.makedirs(viz_path)

    #     for f in range(num_frames):
    #         pF = (Fs[0,:,f].permute(1,2,0).numpy() * 255.).astype(np.uint8)
    #         pE = pred[f]
    #         canvas = overlay_davis(pF, pE, palette)
    #         canvas = Image.fromarray(canvas)
    #         canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))

    #     vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
    #     frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')
    #     os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
