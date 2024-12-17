"""
Use Dpm_solver to generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import sys

sys.path.append(os.getcwd())

import torch
from torchvision import utils

import numpy as np
import torch as th
import torch.distributed as dist

from entropy_driven_guided_diffusion import dist_util, logger
from entropy_driven_guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    num_count = 0
    category_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                     'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                     'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker',
                     'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
                     'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver',
                     'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush',
                     'Toys', 'Trash_Can', 'Webcam']
    # category_list = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
    #  'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
    #  'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
    #  'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
    # category_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
    #            'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    # while len(all_images) * args.batch_size < args.num_samples:
    from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
    # for category in range(args.num_classes):
    for category in reversed(range(args.num_classes)):
        print(category, category_list[category])
        while num_count * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=category, high=category + 1, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            print(model_kwargs)
            ## 1. Define the noise schedule.
            # noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=th.linspace(1e-4, 0.02, 1000).double())
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=model_kwargs,
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
            shape = args.batch_size, 3, args.image_size, args.image_size
            x_T = th.randn(shape, device=dist_util.dev())
            sample = dpm_solver.sample(
                x_T,
                steps=50,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )

            for i in range(args.batch_size):
                filepath = args.log_dir + '/' + category_list[category]
                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                out_path = args.log_dir + '/' + category_list[
                    category] + '/' + f"{str(category)}_{str(num_count * args.batch_size + i)}.png"
                utils.save_image(
                    sample[i].unsqueeze(0),
                    out_path,
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
            num_count = num_count + 1
        num_count = 0

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=12,
        use_ddim=False,
        model_path="",
        log_dir="",

        sample_suffix="",
        fix_seed=False,
        save_imgs_for_visualization=True,
        specified_class=None,
        detail=False,
        num_classes=1000
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
