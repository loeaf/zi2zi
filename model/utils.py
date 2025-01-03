# -*- coding: utf-8 -*-
import os
import glob
import imageio
import numpy as np
from io import BytesIO
from PIL import Image


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = np.array(Image.open(img)).astype(np.float32)  # float를 float32로 변경
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source
    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    img_pil = Image.fromarray(img.astype('uint8'))
    enlarged = np.array(img_pil.resize((nw, nh), Image.Resampling.BILINEAR))
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    Image.fromarray((concated * 255).astype('uint8')).save(img_path)  # 값 범위 조정


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)

    def resize_image(f):
        img = Image.open(f)
        w, h = img.size
        return np.array(img.resize((int(w * 0.33), int(h * 0.33)), Image.Resampling.NEAREST))

    images = [resize_image(f) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file