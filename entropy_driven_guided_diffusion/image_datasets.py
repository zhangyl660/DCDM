import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import datasets

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            # if self.device is not None:
            #     data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            # if self.device is not None:
            #     data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

def load_domain_data(
    *,
    data_dir,
    image_size,
    random_crop=False,
    random_flip=True,
    domain_id=None,
    dataset_type='imagenet1000'
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None

    # Assume classes are the first part of the filename,
    # before an underscore.
    if dataset_type == 'imagenet1000' or dataset_type == 'officeHome':
        class_names = [os.path.dirname(path).split("/")[-1] for path in all_files]
    elif dataset_type == 'cifar10':
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
    else:
        raise NotImplementedError
    if dataset_type == 'officeHome':
        sorted_classes = {x: i for i, x in enumerate(sorted(bf.listdir(data_dir)))}
    else:
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        domain_id=domain_id,
    )
    return dataset

def load_data_specific(
        *,
        data_dir,
        dataset_name,
        domain,
        image_size,
        class_cond=False,
        random_crop=False,
        random_flip=True,
        domain_cond=False,
        domain_id=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    :param data_dir: a dataset directory.
    :param dataset_name: the dataset name the data want to load
    :param domain: the domain of the given dataset that the data want to load
    :param image_size: the size to which images are resized.
    :param class_cond:
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param domain_cond:
    :param domain_id:
    :return:
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    assert dataset_name in datasets.__dict__, "The given dataset must be code as a class."

    def target_transform(target):
        out_dict = {}
        out_dict['y'] = target
        if domain_cond:
            out_dict['domain'] = domain_id
        return out_dict

    def transform(img):
        if random_crop:
            arr = random_crop_arr(img, image_size)
        else:
            arr = center_crop_arr(img, image_size)

        if random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        return np.transpose(arr, [2, 0, 1])

    specificDataset = datasets.__dict__[dataset_name]
    dataset = specificDataset(root=data_dir, task=domain, download=True, transform=transform, target_transform=target_transform)

    return dataset


def load_data(
    *,
    data_dir,
    image_size,
    class_cond=False,
    random_crop=False,
    random_flip=True,
    dataset_type='imagenet1000'
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        if dataset_type == 'imagenet1000' or dataset_type == 'officeHome':
            class_names = [os.path.dirname(path).split("/")[-1] for path in all_files]
            # print('class_names')
            # print(class_names)
        elif dataset_type == 'cifar10':
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
        else:
            raise NotImplementedError
        if dataset_type == 'officeHome':
            sorted_classes = {x: i for i, x in enumerate(sorted(bf.listdir(data_dir)))}
        else:
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}

        # sorted_classes = {'The_Eiffel_Tower': 0, 'The_Great_Wall_of_China': 1, 'aircraft_carrier': 2, 'alarm_clock': 3, 'ant': 4, 'anvil': 5, 'asparagus': 6, 'axe': 7, 'banana': 8, 'basket': 9, 'bathtub': 10, 'bear': 11, 'bee': 12, 'bird': 13, 'blackberry': 14, 'blueberry': 15, 'bottlecap': 16, 'broccoli': 17, 'bus': 18, 'butterfly': 19, 'cactus': 20, 'cake': 21, 'calculator': 22, 'camel': 23, 'camera': 24, 'candle': 25, 'cannon': 26, 'canoe': 27, 'carrot': 28, 'castle': 29, 'cat': 30, 'ceiling_fan': 31, 'cell_phone': 32, 'cello': 33, 'chair': 34, 'chandelier': 35, 'coffee_cup': 36, 'compass': 37, 'computer': 38, 'cow': 39, 'crab': 40, 'crocodile': 41, 'cruise_ship': 42, 'dog': 43, 'dolphin': 44, 'dragon': 45, 'drums': 46, 'duck': 47, 'dumbbell': 48, 'elephant': 49, 'eyeglasses': 50, 'feather': 51, 'fence': 52, 'fish': 53, 'flamingo': 54, 'flower': 55, 'foot': 56, 'fork': 57, 'frog': 58, 'giraffe': 59, 'goatee': 60, 'grapes': 61, 'guitar': 62, 'hammer': 63, 'helicopter': 64, 'helmet': 65, 'horse': 66, 'kangaroo': 67, 'lantern': 68, 'laptop': 69, 'leaf': 70, 'lion': 71, 'lipstick': 72, 'lobster': 73, 'microphone': 74, 'monkey': 75, 'mosquito': 76, 'mouse': 77, 'mug': 78, 'mushroom': 79, 'onion': 80, 'panda': 81, 'peanut': 82, 'pear': 83, 'peas': 84, 'pencil': 85, 'penguin': 86, 'pig': 87, 'pillow': 88, 'pineapple': 89, 'potato': 90, 'power_outlet': 91, 'purse': 92, 'rabbit': 93, 'raccoon': 94, 'rhinoceros': 95, 'rifle': 96, 'saxophone': 97, 'screwdriver': 98, 'sea_turtle': 99, 'see_saw': 100, 'sheep': 101, 'shoe': 102, 'skateboard': 103, 'snake': 104, 'speedboat': 105, 'spider': 106, 'squirrel': 107, 'strawberry': 108, 'streetlight': 109, 'string_bean': 110, 'submarine': 111, 'swan': 112, 'table': 113, 'teapot': 114, 'teddy-bear': 115, 'television': 116, 'tiger': 117, 'toe': 118, 'train': 119, 'truck': 120, 'umbrella': 121, 'vase': 122, 'watermelon': 123, 'whale': 124, 'zebra': 125}

        # sorted_classes = {'The_Eiffel_Tower': 0, 'The_Great_Wall_of_China': 1, 'The_Mona_Lisa': 2, 'aircraft_carrier': 3, 'airplane': 4, 'alarm_clock': 5, 'ambulance': 6, 'angel': 7, 'animal_migration': 8, 'ant': 9, 'anvil': 10, 'apple': 11, 'arm': 12, 'asparagus': 13, 'axe': 14, 'backpack': 15, 'banana': 16, 'bandage': 17, 'barn': 18, 'baseball': 19, 'baseball_bat': 20, 'basket': 21, 'basketball': 22, 'bat': 23, 'bathtub': 24, 'beach': 25, 'bear': 26, 'beard': 27, 'bed': 28, 'bee': 29, 'belt': 30, 'bench': 31, 'bicycle': 32, 'binoculars': 33, 'bird': 34, 'birthday_cake': 35, 'blackberry': 36, 'blueberry': 37, 'book': 38, 'boomerang': 39, 'bottlecap': 40, 'bowtie': 41, 'bracelet': 42, 'brain': 43, 'bread': 44, 'bridge': 45, 'broccoli': 46, 'broom': 47, 'bucket': 48, 'bulldozer': 49, 'bus': 50, 'bush': 51, 'butterfly': 52, 'cactus': 53, 'cake': 54, 'calculator': 55, 'calendar': 56, 'camel': 57, 'camera': 58, 'camouflage': 59, 'campfire': 60, 'candle': 61, 'cannon': 62, 'canoe': 63, 'car': 64, 'carrot': 65, 'castle': 66, 'cat': 67, 'ceiling_fan': 68, 'cell_phone': 69, 'cello': 70, 'chair': 71, 'chandelier': 72, 'church': 73, 'circle': 74, 'clarinet': 75, 'clock': 76, 'cloud': 77, 'coffee_cup': 78, 'compass': 79, 'computer': 80, 'cookie': 81, 'cooler': 82, 'couch': 83, 'cow': 84, 'crab': 85, 'crayon': 86, 'crocodile': 87, 'crown': 88, 'cruise_ship': 89, 'cup': 90, 'diamond': 91, 'dishwasher': 92, 'diving_board': 93, 'dog': 94, 'dolphin': 95, 'donut': 96, 'door': 97, 'dragon': 98, 'dresser': 99, 'drill': 100, 'drums': 101, 'duck': 102, 'dumbbell': 103, 'ear': 104, 'elbow': 105, 'elephant': 106, 'envelope': 107, 'eraser': 108, 'eye': 109, 'eyeglasses': 110, 'face': 111, 'fan': 112, 'feather': 113, 'fence': 114, 'finger': 115, 'fire_hydrant': 116, 'fireplace': 117, 'firetruck': 118, 'fish': 119, 'flamingo': 120, 'flashlight': 121, 'flip_flops': 122, 'floor_lamp': 123, 'flower': 124, 'flying_saucer': 125, 'foot': 126, 'fork': 127, 'frog': 128, 'frying_pan': 129, 'garden': 130, 'garden_hose': 131, 'giraffe': 132, 'goatee': 133, 'golf_club': 134, 'grapes': 135, 'grass': 136, 'guitar': 137, 'hamburger': 138, 'hammer': 139, 'hand': 140, 'harp': 141, 'hat': 142, 'headphones': 143, 'hedgehog': 144, 'helicopter': 145, 'helmet': 146, 'hexagon': 147, 'hockey_puck': 148, 'hockey_stick': 149, 'horse': 150, 'hospital': 151, 'hot_air_balloon': 152, 'hot_dog': 153, 'hot_tub': 154, 'hourglass': 155, 'house': 156, 'house_plant': 157, 'hurricane': 158, 'ice_cream': 159, 'jacket': 160, 'jail': 161, 'kangaroo': 162, 'key': 163, 'keyboard': 164, 'knee': 165, 'knife': 166, 'ladder': 167, 'lantern': 168, 'laptop': 169, 'leaf': 170, 'leg': 171, 'light_bulb': 172, 'lighter': 173, 'lighthouse': 174, 'lightning': 175, 'line': 176, 'lion': 177, 'lipstick': 178, 'lobster': 179, 'lollipop': 180, 'mailbox': 181, 'map': 182, 'marker': 183, 'matches': 184, 'megaphone': 185, 'mermaid': 186, 'microphone': 187, 'microwave': 188, 'monkey': 189, 'moon': 190, 'mosquito': 191, 'motorbike': 192, 'mountain': 193, 'mouse': 194, 'moustache': 195, 'mouth': 196, 'mug': 197, 'mushroom': 198, 'nail': 199, 'necklace': 200, 'nose': 201, 'ocean': 202, 'octagon': 203, 'octopus': 204, 'onion': 205, 'oven': 206, 'owl': 207, 'paint_can': 208, 'paintbrush': 209, 'palm_tree': 210, 'panda': 211, 'pants': 212, 'paper_clip': 213, 'parachute': 214, 'parrot': 215, 'passport': 216, 'peanut': 217, 'pear': 218, 'peas': 219, 'pencil': 220, 'penguin': 221, 'piano': 222, 'pickup_truck': 223, 'picture_frame': 224, 'pig': 225, 'pillow': 226, 'pineapple': 227, 'pizza': 228, 'pliers': 229, 'police_car': 230, 'pond': 231, 'pool': 232, 'popsicle': 233, 'postcard': 234, 'potato': 235, 'power_outlet': 236, 'purse': 237, 'rabbit': 238, 'raccoon': 239, 'radio': 240, 'rain': 241, 'rainbow': 242, 'rake': 243, 'remote_control': 244, 'rhinoceros': 245, 'rifle': 246, 'river': 247, 'roller_coaster': 248, 'rollerskates': 249, 'sailboat': 250, 'sandwich': 251, 'saw': 252, 'saxophone': 253, 'school_bus': 254, 'scissors': 255, 'scorpion': 256, 'screwdriver': 257, 'sea_turtle': 258, 'see_saw': 259, 'shark': 260, 'sheep': 261, 'shoe': 262, 'shorts': 263, 'shovel': 264, 'sink': 265, 'skateboard': 266, 'skull': 267, 'skyscraper': 268, 'sleeping_bag': 269, 'smiley_face': 270, 'snail': 271, 'snake': 272, 'snorkel': 273, 'snowflake': 274, 'snowman': 275, 'soccer_ball': 276, 'sock': 277, 'speedboat': 278, 'spider': 279, 'spoon': 280, 'spreadsheet': 281, 'square': 282, 'squiggle': 283, 'squirrel': 284, 'stairs': 285, 'star': 286, 'steak': 287, 'stereo': 288, 'stethoscope': 289, 'stitches': 290, 'stop_sign': 291, 'stove': 292, 'strawberry': 293, 'streetlight': 294, 'string_bean': 295, 'submarine': 296, 'suitcase': 297, 'sun': 298, 'swan': 299, 'sweater': 300, 'swing_set': 301, 'sword': 302, 'syringe': 303, 't-shirt': 304, 'table': 305, 'teapot': 306, 'teddy-bear': 307, 'telephone': 308, 'television': 309, 'tennis_racquet': 310, 'tent': 311, 'tiger': 312, 'toaster': 313, 'toe': 314, 'toilet': 315, 'tooth': 316, 'toothbrush': 317, 'toothpaste': 318, 'tornado': 319, 'tractor': 320, 'traffic_light': 321, 'train': 322, 'tree': 323, 'triangle': 324, 'trombone': 325, 'truck': 326, 'trumpet': 327, 'umbrella': 328, 'underwear': 329, 'van': 330, 'vase': 331, 'violin': 332, 'washing_machine': 333, 'watermelon': 334, 'waterslide': 335, 'whale': 336, 'wheel': 337, 'windmill': 338, 'wine_bottle': 339, 'wine_glass': 340, 'wristwatch': 341, 'yoga': 342, 'zebra': 343, 'zigzag': 344}

        sorted_classes = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5, 'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10, 'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15, 'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20, 'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25, 'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30, 'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35, 'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41, 'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46, 'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51, 'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57, 'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62, 'Trash_Can': 63, 'Webcam': 64}

        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    return dataset


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        domain_id=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.domain_id = domain_id

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.domain_id is not None:
            out_dict["domain"] = np.array(self.domain_id, dtype=np.int64)
            #out_dict["domain"] = self.domain_id
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
