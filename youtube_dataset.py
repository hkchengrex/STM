import os
from os import path
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob

class YOUTUBE_VOS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, all_frames_root, id, total_id):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations')
        self.image_dir = path.join(root, 'JPEGImages')
        self.all_frames_image_dir = path.join(all_frames_root, 'JPEGImages')
        self.videos = []
        self.num_skip_frames = {}
        self.num_frames = {}
        self.shape = {}

        self_vid_list = sorted(os.listdir(self.image_dir))
        
        start_idx = len(self_vid_list) * id // total_id
        end_idx = len(self_vid_list) * (id+1) // total_id

        print('This process handles video %d to %d out of %d' % (start_idx, end_idx-1, len(self_vid_list)))

        self_vid_list = self_vid_list[start_idx:end_idx]

        for vid in self_vid_list:
            self.videos.append(vid)
            self.num_skip_frames[vid] = len(os.listdir(os.path.join(self.image_dir, vid)))
            self.num_frames[vid] = len(os.listdir(os.path.join(self.all_frames_image_dir, vid)))
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        self.K = 11

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['num_skip_frames'] = self.num_skip_frames[video]
        info['num_objects'] = 0

        N_all_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)

        N_frames = np.empty((self.num_skip_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_skip_frames[video],)+self.shape[video], dtype=np.uint8)
        for i, f in enumerate(sorted(os.listdir(path.join(self.image_dir, video)))):
            img_file = path.join(self.image_dir, video, f)
            N_frames[i] = np.array(Image.open(img_file).convert('RGB'))/255.
 
            mask_file = path.join(self.mask_dir, video, f.replace('.jpg', '.png'))
            N_masks[i] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

            info['num_objects'] = max(info['num_objects'], N_masks[i].max())

        for i, f in enumerate(sorted(os.listdir(path.join(self.all_frames_image_dir, video)))):
            img_file = path.join(self.all_frames_image_dir, video, f)
            N_all_frames[i] = np.array(Image.open(img_file).convert('RGB'))/255.

        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()

        AFs = torch.from_numpy(np.transpose(N_all_frames.copy(), (3, 0, 1, 2)).copy()).float()
        
        return Fs, Ms, AFs, info



if __name__ == '__main__':
    pass
