import os
from os import path
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

from extra_para_io import load_sub_val
import glob

class YOUTUBE_VOS_MO_Test_val(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, all_frames_root, fz_root, start_idx, end_idx):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations')
        self.image_dir = path.join(all_frames_root, 'JPEGImages')
        self.fz_mask_dir = path.join(fz_root, 'Annotations')
        self.videos = []
        self.num_frames = {}
        self.shape = {}
        self.real_shape = {}
        self.frames_name = {}

        self_vid_list = sorted(os.listdir(self.image_dir))
        sub_val_list = load_sub_val()
        self_vid_list = [v for v in self_vid_list if v in sub_val_list]

        print('This process handles video %d to %d out of %d' % (start_idx, end_idx-1, len(self_vid_list)))

        self_vid_list = self_vid_list[start_idx:end_idx]

        for vid in self_vid_list:

            self.videos.append(vid)
            self.num_frames[vid] = len(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames_name[vid] = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

            first_mask = os.listdir(path.join(self.fz_mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.fz_mask_dir, vid, first_mask)).convert("P"))
            self.real_shape[vid] = np.shape(_mask)

        self.K = 7

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
        info['num_objects'] = 0
        info['frames_name'] = self.frames_name[video]
        info['real_shape'] = self.real_shape[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        # N_masks = np.empty((self.num_skip_frames[video],)+self.shape[video], dtype=np.uint8)
        N_ref_msk = []
        ref_id = []

        # Store the new object id presented in the reference mask
        ref_new_obj = []
        # Store all the existing objects
        existing_obj = []

        for i, f in enumerate(sorted(os.listdir(path.join(self.image_dir, video)))):
            img_file = path.join(self.image_dir, video, f)
            N_frames[i] = np.array(Image.open(img_file).convert('RGB'))/255.
 
            mask_file = path.join(self.mask_dir, video, f.replace('.jpg', '.png'))
            if os.path.exists(mask_file):
                msk_array = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                msk_obj = np.unique(msk_array)
                new_obj = np.setdiff1d(msk_obj, existing_obj)

                if len(new_obj) > 0:
                    # Neglect the objects that are not new
                    for o in existing_obj:
                        if o != 0:
                            msk_array[msk_array==o] = 0
                    # Update existing objects
                    existing_obj = np.concatenate((existing_obj, new_obj))
                    N_ref_msk.append(msk_array)
                    ref_id.append(i)
                    ref_new_obj.append(new_obj)

        N_ref_msk = np.array(N_ref_msk)
        info['num_objects'] = N_ref_msk.max()
        info['ref_id'] = ref_id
        info['ref_new_obj'] = ref_new_obj

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_ref_msk).copy()).float()
        
        return Fs, Ms, info



if __name__ == '__main__':
    pass
