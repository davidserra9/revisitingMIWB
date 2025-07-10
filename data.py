import torch
import glob
import json
import logging

import torchvision.transforms
from PIL import Image
import numpy as np
import os
import torchvision.transforms.functional as TF
import random

class WBDataset(torch.utils.data.Dataset):
    def __init__(self, input_root, target_root, places_list, transform=None):
        """ Dataset class for the LSMI dataset
        :param input_root: path to the input images
        :param target_root: path to the target images
        :param places_list: list of places to be loaded to this split
        :param transform: torchvision transform to be applied to the images
        """
        self.input_root = input_root
        self.target_root = target_root
        self.places_list = places_list
        self.transform = transform

        self.input_images = []
        self.target_images = []
        self.color_chart_areas = []

        for place in places_list:

            # Input images
            place_dict = {}
            place_img_paths = glob.glob(os.path.join(input_root, place + '_*'))
            for img_path in place_img_paths:
                if img_path.split('/')[-1].split('_')[0] + '_' + img_path.split('/')[-1].split('_')[1] in place_dict.keys():
                    place_dict[img_path.split('/')[-1].split('_')[0] + '_' + img_path.split('/')[-1].split('_')[1]] += [img_path]

                else:
                    place_dict.update({img_path.split('/')[-1].split('_')[0] + '_' + img_path.split('/')[-1].split('_')[1]: [img_path]})

            for name, img_path_list in place_dict.items():
                place_illuminant_dict = {}
                for img_path in img_path_list:
                    if img_path.split('/')[-1].split('_')[2] == 'C':
                        place_illuminant_dict.update({'cloudy': img_path})
                    elif img_path.split('/')[-1].split('_')[2] == 'D':
                        place_illuminant_dict.update({'daylight': img_path})
                    elif img_path.split('/')[-1].split('_')[2] == 'F':
                        place_illuminant_dict.update({'fluorescent': img_path})
                    elif img_path.split('/')[-1].split('_')[2] == 'T':
                        place_illuminant_dict.update({'tungsten': img_path})
                    elif img_path.split('/')[-1].split('_')[2] == 'S':
                        place_illuminant_dict.update({'shade': img_path})
                    else:
                        logging.info('Wrong image type. It must be C, D, F, T or S')

                assert len(place_illuminant_dict) == 5, 'There must be 5 images per place'
                self.input_images.append(place_illuminant_dict)

                # Target image
                if len(glob.glob(os.path.join(target_root, name + '_*'))) != 1:
                    logging.error('There must be only one target image per place')

                target_img_path = glob.glob(os.path.join(target_root, name + '_*'))[0]
                color_chart_area = int(target_img_path.split('/')[-1].split('_')[-1].split('.')[0]) if len(target_img_path.split('/')[-1].split('_')) == 5 else 0

                self.color_chart_areas.append(color_chart_area)
                self.target_images.append(glob.glob(os.path.join(target_root, name + '_*'))[0])

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        images_dict = {}
        input_images = self.input_images[idx]
        target_image = self.target_images[idx]

        for type, img_path in input_images.items():
            if self.transform is not None:
                img = Image.open(img_path).convert('RGB')
                images_dict.update({type: self.transform(TF.to_tensor(img))})
            else:
                img = Image.open(img_path).convert('RGB')
                images_dict.update({type: TF.to_tensor(img)})

        if self.transform is not None:
            img = Image.open(target_image).convert('RGB')
            images_dict.update({'target': self.transform(TF.to_tensor(img))})
        else:
            img = Image.open(target_image).convert('RGB')
            images_dict.update({'target': TF.to_tensor(img)})

        images_dict.update({'name': target_image.split('/')[-1].split('_')[0] + '_' + target_image.split('/')[-1].split('_')[1]})
        images_dict.update({'color_chart_area': self.color_chart_areas[idx]})

        return images_dict

class AfifiAWBDataset(torch.utils.data.Dataset):
    def __init__(self, images_root, transform=None):
        """ Dataset class for the LSMI dataset
        :param input_root: path to the input images
        :param target_root: path to the target images
        :param places_list: list of places to be loaded to this split
        :param transform: torchvision transform to be applied to the images
        """
        self.images_root = images_root
        self.transform = transform

        all_scenes = [scene.split('/')[-1].split('_')[0] + '_' + scene.split('/')[-1].split('_')[1] for scene in glob.glob(os.path.join(images_root, '*'))]
        self.scenes = list(set(all_scenes))

        self.images = []
        for scene in self.scenes:
            scene_dict = {}
            for img_path in glob.glob(os.path.join(images_root, scene + '_*')):
                if img_path.split('/')[-1].split('_')[2] == 'C':
                    scene_dict.update({'cloudy': img_path})
                elif img_path.split('/')[-1].split('_')[2] == 'D':
                    scene_dict.update({'daylight': img_path})
                elif img_path.split('/')[-1].split('_')[2] == 'F':
                    scene_dict.update({'fluorescent': img_path})
                elif img_path.split('/')[-1].split('_')[2] == 'T':
                    scene_dict.update({'tungsten': img_path})
                elif img_path.split('/')[-1].split('_')[2] == 'S':
                    scene_dict.update({'shade': img_path})
                elif img_path.split('/')[-1].split('_')[2] == 'G':
                    scene_dict.update({'target': img_path})
                else:
                    logging.info('Wrong image type. It must be C, D, F, T, S or G')

            assert len(scene_dict) == 6, 'There must be 6 images per scene C, D, F, T, S or G (ground truth)'
            self.images.append(scene_dict)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return_dict = {}
    
        for t, img_path in self.images[idx].items():
            if self.transform is not None:
                return_dict.update({t: self.transform(TF.to_tensor(Image.open(img_path).convert('RGB')))})
            else:
                return_dict.update({t: TF.to_tensor(Image.open(img_path).convert('RGB'))})

        return_dict.update({'name': self.images[idx]['target'].split('/')[-1].split('_')[0] + '_' + self.images[idx]['target'].split('/')[-1].split('_')[1]})
        return_dict.update({'color_chart_area': 0})

        return return_dict

class ColorConstancyDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, transform=None):
        """ Dataset class for the LSMI dataset
        :param input_root: path to the input images
        :param target_root: path to the target images
        :param places_list: list of places to be loaded to this split
        :param transform: torchvision transform to be applied to the images
        """
        self.image_root = image_root
        self.transform = transform

        self.images = []
        for target_image in glob.glob(os.path.join(image_root, '*_G_AS.png')):
            scene_dict = {}

            for img_type in ['C_CS', 'D_CS', 'F_CS', 'T_CS', 'S_CS']:
                input_image_path = os.path.join(image_root, target_image.split('/')[-1].replace('G_AS', img_type))

                # Check if file exists
                if not os.path.isfile(input_image_path):
                    input_image_path = input_image_path.replace('CS.png', 'AS.png')
                if not os.path.isfile(input_image_path):
                    logging.info('File not found: {}'.format(input_image_path))
                    # skip this image
                    continue

                if img_type == 'C_CS':
                    scene_dict.update({'cloudy': input_image_path})
                elif img_type == 'D_CS':
                    scene_dict.update({'daylight': input_image_path})
                elif img_type == 'F_CS':
                    scene_dict.update({'fluorescent': input_image_path})
                elif img_type == 'T_CS':
                    scene_dict.update({'tungsten': input_image_path})
                elif img_type == 'S_CS':
                    scene_dict.update({'shade': input_image_path})
                else:
                    logging.info('Wrong image type. It must be C, D, F, T, S')

            scene_dict.update({'target': target_image})

            if len(scene_dict) != 6:
                logging.info('Wrong number of images in scene {}'.format(target_image))
                continue

            self.images.append(scene_dict)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return_dict = {}

        for type, img_path in self.images[idx].items():
            if self.transform is not None:
                return_dict.update({type: self.transform(TF.to_tensor(Image.open(img_path).convert('RGB')))})
            else:
                return_dict.update({type: TF.to_tensor(Image.open(img_path).convert('RGB'))})

        return_dict.update({'name': self.images[idx]['target'].split('/')[-1].split('_')[0] + '_' + self.images[idx]['target'].split('/')[-1].split('_')[1]})
        return_dict.update({'color_chart_area': 0})

        return return_dict

class CubePlusDataset(torch.utils.data.Dataset):
    def __init__(self, input_root, target_root, transform=None, num_images=None):
        """ Dataset class for the LSMI dataset
        :param input_root: path to the input images
        :param target_root: path to the target images
        :param places_list: list of places to be loaded to this split
        :param transform: torchvision transform to be applied to the images
        """
        self.input_root = input_root
        self.target_root = target_root
        self.transform = transform
        self.num_images = num_images

        self.images = []
        for target_image in glob.glob(os.path.join(target_root, '*')):
            scene_dict = {}

            for img_type in ['_C', '_D', '_F', '_T', '_S']:
                input_image_path = os.path.join(input_root, target_image.split('/')[-1].replace('.JPG', f'{img_type}.JPG'))

                if not os.path.isfile(input_image_path):
                    logging.info('File not found: {}'.format(input_image_path))
                    # skip this image
                    continue

                if img_type == '_C':
                    scene_dict.update({'cloudy': input_image_path})
                elif img_type == '_D':
                    scene_dict.update({'daylight': input_image_path})
                elif img_type == '_F':
                    scene_dict.update({'fluorescent': input_image_path})
                elif img_type == '_T':
                    scene_dict.update({'tungsten': input_image_path})
                elif img_type == '_S':
                    scene_dict.update({'shade': input_image_path})
                else:
                    logging.info('Wrong image type. It must be C, D, F, T, S')

            scene_dict.update({'target': target_image})

            if len(scene_dict) != 6:
                logging.info('Wrong number of images in scene {}'.format(target_image))
                continue

            self.images.append(scene_dict)

    def __len__(self):
        return self.num_images if self.num_images is not None else len(self.images)

    def __getitem__(self, idx):
        return_dict = {}

        if self.num_images is not None:
            idx = random.randint(0, len(self.images) - 1)

        images = self.images[idx]

        for type, img_path in images.items():
            if self.transform is not None:
                return_dict.update({type: self.transform(TF.to_tensor(Image.open(img_path).convert('RGB')))})
            else:
                return_dict.update({type: TF.to_tensor(Image.open(img_path).convert('RGB'))})

        return_dict.update({'name': images['target'].split('/')[-1]})
        return_dict.update({'color_chart_area': 0})

        return return_dict


def create_datasets(data_args):
    """ Create training, validation and test datasets
    :param data_args: dictionary or list of dictionaries with the arguments for each dataset. The dictionary has to have
    the following format:
    {'input_root': path to the input images,
     'target_root': path to the target images,
     'splits': path to the json split file}

    :return: training, validation and test datasets. If there is no validation, validation dataset is None
    """

    if not isinstance(data_args, list):
        data_args = [data_args]
        train_transforms = None

    # If both datasets are joined, nikon images have to be resized to 500x700 (sony size)
    # Only for speed-up training. On validation and test have the original size (batch size is 1)
    elif len(data_args) > 1:
        train_transforms = torchvision.transforms.Resize((500, 700), antialias=True)

    else:
        train_transforms = None

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for arg in data_args:
        with open(arg['splits'], 'r') as f:
            f = json.load(f)

        if 'two_illum_train' in list(f.keys()):
            # Load nikon_meta.json and sony_meta.json
            train_places_list = []
            valid_places_list = []
            test_places_list = []

            for key, place_list in f.items():
                if 'train' in key:
                    train_places_list += place_list
                elif 'val' in key:
                    valid_places_list += place_list
                elif 'test' in key:
                    test_places_list += place_list

        elif 'Xtrain' in list(f.keys()):
            train_places_list = [p.split('_')[0] + '_' + p.split('_')[1] for p in f['Xtrain']]
            valid_places_list = [p.split('_')[0] + '_' + p.split('_')[1] for p in f['Xvalid']] if 'Xvalid' in list(f.keys()) else None
            test_places_list = [p.split('_')[0] + '_' + p.split('_')[1] for p in f['Xtest']]

        else:
            logging.info('Wrong split file')
            exit()

        if 'valid_input_root' not in list(arg.keys()):
            arg['valid_input_root'] = arg['input_root']
        if 'valid_target_root' not in list(arg.keys()):
            arg['valid_target_root'] = arg['target_root']

        train_datasets.append(WBDataset(input_root=arg['input_root'],
                                        target_root=arg['target_root'],
                                        places_list=train_places_list,
                                        transform=train_transforms))
        
        if valid_places_list is not None and len(valid_places_list) > 0:
            val_datasets.append(WBDataset(input_root=arg['valid_input_root'],
                                          target_root=arg['valid_target_root'],
                                          places_list=valid_places_list))

        test_datasets.append(WBDataset(input_root=arg['valid_input_root'],
                                       target_root=arg['valid_target_root'],
                                       places_list=test_places_list))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets) if len(val_datasets) > 0 else None
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    logging.info(f"Train dataset created with {len(train_dataset)} samples")
    logging.info(f"Validation dataset created with {len(val_dataset)} samples" if val_dataset is not None else "No validation dataset")
    logging.info(f"Test dataset created with {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset