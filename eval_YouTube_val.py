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
from youtube_dataset_val import YOUTUBE_VOS_MO_Test_val
from model import STM


torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    # parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("--vos_root", type=str, help="path to data", default='../YouTube/vos/valid_480p')
    parser.add_argument("--af_root", type=str, help="path to data", default='../YouTube/vos/all_frames/valid_480p')
    parser.add_argument("--fz_root", type=str, help="path to full sized data", default='../YouTube/vos/valid')

    parser.add_argument("--id", type=int, help='Id out of total ID for partitioning', default=0)
    parser.add_argument("--start", type=int, help='Start IDX  (inclusive)', default=0)
    parser.add_argument("--end", type=int, help='END IDX (inclusive)', default=0)

    parser.add_argument("--extra_id", type=str, default='')

    return parser.parse_args()

args = get_arguments()

# GPU = args.g
VIZ = args.viz
VOS_ROOT = args.vos_root
AF_ROOT = args.af_root
FZ_ROOT = args.fz_root

id = args.id
start_idx = args.start
end_idx = args.end

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on YouTube')

# os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

palette = Image.open(path.join(VOS_ROOT, 'Annotations/0a49f5265b/00000.png')).getpalette()

def Run_video(Fs, Ms, ref_id, num_objects, real_shape, Mem_every=None, Mem_number=None):

    # print(Fs.shape, Ms.shape, AFs.shape)
    b, _, t, h, w = Fs.shape
    _, k, _, _, _ = Ms.shape
    rh, rw = real_shape

    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros((b, k, t, h, w), dtype=torch.float32, device=Ms.device)
    ref_key = []
    ref_value = []
    for ei, i in enumerate(ref_id):
        Es[:,:,i] = Ms[:,:,ei]
        k, v = model(Fs[:,:,i], Es[:,:,i], torch.tensor([num_objects]))
        ref_key.append(k)
        ref_value.append(v)

    keys = torch.cat(ref_key, 3)
    values = torch.cat(ref_value, 3)

    for t_step in range(t):
        if t_step in ref_id:
            continue

        if (t_step+1) not in ref_id:
            # memorize previous frame
            prev_key, prev_value = model(Fs[:,:,t_step-1], Es[:,:,t_step-1], torch.tensor([num_objects]))
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        else:
            this_keys = keys
            this_values = values
        
        # segment
        logit = model(Fs[:,:,t_step], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t_step] = F.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        else:
            del this_keys
            del this_values

    up_Es = F.interpolate(Es, size=(t, rh, rw), mode='trilinear', align_corners=False)
        
    pred = np.argmax(up_Es[0].numpy(), axis=0).astype(np.uint8)
    return pred, up_Es



Testset = YOUTUBE_VOS_MO_Test_val(VOS_ROOT, AF_ROOT, FZ_ROOT, start_idx=start_idx, end_idx=end_idx)
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = 'YouTube_val_%d_%d' % (start_idx, end_idx)
skipped = []

for seq, V in progressbar(enumerate(Testloader), max_value=len(Testloader), redirect_stdout=True):
    Fs, Ms, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    num_objects = info['num_objects'][0]
    frames_name = info['frames_name']
    real_shape = info['real_shape']
    ref_id = info['ref_id']
    ref_id = [r[0] for r in ref_id] # Unsqueeze the batch dim

    print(seq_name)
    
    with torch.no_grad():
        try:
            try:
                pred, Es = Run_video(Fs, Ms, ref_id, num_objects, real_shape, Mem_every=5, Mem_number=None)
            except:
                print('Mem 5 failed. ')
                try:
                    pred, Es = Run_video(Fs, Ms, ref_id, num_objects, real_shape, Mem_every=7, Mem_number=None)
                except:
                    print('Mem 7 failed. ')
                    pred, Es = Run_video(Fs, Ms, ref_id, num_objects, real_shape, Mem_every=10, Mem_number=None)


            # Save results for quantitative eval ######################
            test_path = os.path.join('./test', code_name, seq_name)
            if not os.path.exists(test_path):
                os.makedirs(test_path)
                
            for i, f in enumerate(frames_name):
                img_E = Image.fromarray(pred[i])
                img_E.putpalette(palette)
                img_E.save(os.path.join(test_path, f[0].replace('.jpg', '.png')))

        except RuntimeError as e:
            print('Exception', e, seq_name)
            skipped.append(seq_name)
            print('Skipped: ', skipped)
        


