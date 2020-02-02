import sys
import os
from os import path

from PIL import Image
import numpy as np
from progressbar import progressbar
from multiprocessing import Pool

new_min_size = 480

def process_fun(vid_name):
    vid_path = path.join(hr_path, vid_name)
    vid_out_path = path.join(out_path, vid_name)
    os.makedirs(vid_out_path, exist_ok=True)

    for im_name in os.listdir(vid_path):
        hr_im = Image.open(path.join(vid_path, im_name))
        w, h = hr_im.size

        ratio = new_min_size / min(w, h)

        lr_im = hr_im.resize((int(w*ratio), int(h*ratio)), Image.BICUBIC)
        lr_im.save(path.join(vid_out_path, im_name))

if __name__ == '__main__':
    in_path = sys.argv[1]

    for split in os.listdir(in_path):
        split_path = path.join(in_path, split)

        out_path = path.join(split_path, 'JPEGImages_480p')
        os.makedirs(out_path, exist_ok=True)
        hr_path = path.join(split_path, 'JPEGImages')

        pool = Pool(processes=16)

        videos = os.listdir(hr_path)
        for _ in progressbar(pool.imap_unordered(process_fun, videos), max_value=len(videos)):
            pass

    print('Done.')