import os
import torch

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2


class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)


def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan', save_dir=str):
    """ Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images):  # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler, save_dir=save_dir)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len


def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan', save_dir=str):
    """ Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function. """

    dir_frames_f = os.path.join(save_dir, 'f')
    os.makedirs(str(dir_frames_f), exist_ok=True)

    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images):  # handle video to images
        print(images)
        # images = load_video_to_cv2(images)

        cmd = f"ffmpeg -loglevel error -i '{images}' {save_dir}/f/%04d.png"
        os.system(cmd)
    # ------------------------ set up GFPGAN restorer ------------------------
    if method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer':  # TODO:
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')

    # determine model paths
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')

    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')

    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None
    )

    images_list = sorted(
        [os.path.join(dir_frames_f, frame) for frame in os.listdir(dir_frames_f) if frame.endswith('.png')]
    )

    # ------------------------ restore ------------------------
    for idx in tqdm(range(len(images_list)), 'Face Enhancer:'):
        img = cv2.imread(images_list[idx])

        cropped_faces, restored_faces, r_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        cv2.imwrite(f'{save_dir}/f-{idx:04d}.png', r_img)

        # r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)

        # yield r_img

    return True
