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
import torch.nn.functional as F

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

    if args.fix_seed:
        seed = 23333 + dist.get_rank()
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

        random.seed(seed)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

    logger.configure(dir=args.log_dir)
    logger.log(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if args.model_path:
        logger.log("loading model from {}".format(args.model_path))
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu"),
            strict=True
        )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    cls_params = args_to_dict(args, classifier_defaults().keys())
    cls_params["num_classes"] = 2
    classifier = create_classifier(**cls_params)
    if args.classifier_path:
        logger.log("loading classifier from {}".format(args.classifier_path))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu"),
            strict=True
        )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, condition, domain=None, **kwargs):
        assert domain is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), domain.view(-1)]
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]
        return cond_grad


    logger.log("sampling...")
    all_images = []
    all_labels = []
    num_count = 0
    # category_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
    #                  'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
    #                  'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
    #                  'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker',
    #                  'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil',
    #                  'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver',
    #                  'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush',
    #                  'Toys', 'Trash_Can', 'Webcam']
    category_list = ['The_Eiffel_Tower', 'The_Great_Wall_of_China','aircraft_carrier', 'alarm_clock', 'ant', 'anvil','asparagus', 'axe', 'banana', 'basket', 'bathtub', 'bear', 'bee', 
               'bird', 'blackberry', 'blueberry', 'bottlecap', 'broccoli', 'bus', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'camera',
               'candle', 'cannon', 'canoe', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'coffee_cup', 'compass', 'computer', 'cow', 'crab', 'crocodile',
               'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck', 'dumbbell', 'elephant','eyeglasses', 'feather', 'fence', 'fish', 'flamingo', 'flower', 'foot', 'fork', 'frog', 'giraffe', 'goatee', 'grapes', 'guitar', 'hammer', 'helicopter', 'helmet', 'horse', 'kangaroo', 'lantern', 'laptop', 'leaf', 'lion', 'lipstick', 'lobster','microphone', 'monkey', 'mosquito', 'mouse', 'mug', 'mushroom', 'onion', 'panda', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'pig', 'pillow', 'pineapple','potato', 'power_outlet', 'purse',
               'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone', 'screwdriver', 'sea_turtle', 'see_saw', 'sheep',
               'shoe', 'skateboard', 'snake', 'speedboat', 'spider', 'squirrel', 'strawberry', 'streetlight', 'string_bean',
               'submarine', 'swan', 'table', 'teapot', 'teddy-bear', 'television', 'tiger','toe', 'train', 'truck', 'umbrella', 'vase', 'watermelon', 'whale', 'zebra']
    # category_list = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
    #                  'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
    #                  'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
    #                  'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser',
    #                  'trash_can']
    # category_list = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
    #            'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
    # category_list = ['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan', 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
    from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
    # for category in range(NUM_CLASSES):
    for category in range(args.num_classes):
    # for category in reversed(range(args.num_classes)):
        print(category, category_list[category])
        while num_count * args.batch_size < args.num_samples:
            model_kwargs = {}
            classifier_kwargs = {}
            if args.specified_domain:
                classes = th.randint(
                    low=category, high=category + 1, size=(args.batch_size,), device=dist_util.dev()
                )
                domains = th.randint(
                    low=int(args.specified_domain), high=int(args.specified_domain) + 1, size=(args.batch_size,),
                    device=dist_util.dev()
                )
            else:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
                )
                domains = th.randint(
                    low=1, high=2, size=(args.batch_size,),
                    device=dist_util.dev()
                )
            model_kwargs["y"] = classes
            classifier_kwargs["domain"] = domains
            # print(model_kwargs)
            ## 1. Define the noise schedule.
            # noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=th.linspace(1e-4, 0.02, 1000).double())
            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=model_kwargs,
                guidance_type="classifier",
                classifier_fn = cond_fn,
                classifier_kwargs= classifier_kwargs,
                guidance_scale= args.classifier_scale,
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
        model_path="",
        log_dir="",
        classifier_path="",
        classifier_scale=1.0,

        fix_seed=False,
        save_imgs_for_visualization=True,
        specified_domain=None,
        detail=False,
        num_classes=1000
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
