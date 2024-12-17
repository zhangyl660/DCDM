"""
Train a diffusion model on images.
"""
import os
import sys
sys.path.append(os.getcwd())


import argparse
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from entropy_driven_guided_diffusion import dist_util, logger
from entropy_driven_guided_diffusion.image_datasets import load_data,ForeverDataIterator
from entropy_driven_guided_diffusion.resample import create_named_schedule_sampler
from entropy_driven_guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from entropy_driven_guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
    logger.log('current rank == {}, total_num = {}'.format(dist.get_rank(), dist.get_world_size()))
    logger.log(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, label_emb=True, map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())
    logger.log("Successfully load pretrain model")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    dataset = ConcatDataset([load_data(
        data_dir=data_dir,
        image_size=args.image_size,
        class_cond=args.class_cond,
        dataset_type='officeHome',
    ) for domain_index, data_dir in enumerate(args.data_dirs)])
    data = ForeverDataIterator(DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True
        ))

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop(args.end_step)


def create_argparser():
    defaults = dict(
        model_path="diffusion/256x256_diffusion.pt",
        end_step=100000,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir="",
        dataset_type='imagenet1000',
        classifier_free=False,
        classifier_free_dropout=0.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, nargs='+')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
