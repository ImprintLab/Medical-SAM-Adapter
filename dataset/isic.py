import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = getattr(args, 'prompt', prompt)
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        mask_array = np.array(mask) / 255
        prompt_data = {}
        if self.prompt in ('click', 'click3'):
            click_count = 1 if self.prompt == 'click' else 3
            foreground = np.argwhere(mask_array > 0)
            if len(foreground) == 0:
                point_label = np.zeros(click_count, dtype=np.int64)
                points = np.column_stack((
                    np.random.randint(0, self.img_size, size=click_count),
                    np.random.randint(0, self.img_size, size=click_count),
                ))
            else:
                selected = np.random.choice(
                    len(foreground), size=click_count, replace=len(foreground) < click_count
                )
                points = foreground[selected][:, ::-1].copy()
                point_label = np.ones(click_count, dtype=np.int64)
            prompt_data.update({'p_label': point_label, 'pt': points})
        elif self.prompt.startswith('bbox'):
            foreground = np.argwhere(mask_array > 0)
            if len(foreground) == 0:
                box = np.array([0, 0, self.img_size - 1, self.img_size - 1], dtype=np.float32)
            else:
                y_min, x_min = foreground.min(axis=0)
                y_max, x_max = foreground.max(axis=0)
                overlap = {'bbox': 1.0, 'bbox0.5': 0.5, 'bbox0.75': 0.75}[self.prompt]
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                lower_scale, upper_scale = 0.0, 1.0
                for _ in range(20):
                    scale = (lower_scale + upper_scale) / 2
                    left = center_x - width * scale / 2
                    right = center_x + width * scale / 2
                    top = center_y - height * scale / 2
                    bottom = center_y + height * scale / 2
                    enclosed = (
                        (foreground[:, 1] >= left)
                        & (foreground[:, 1] <= right)
                        & (foreground[:, 0] >= top)
                        & (foreground[:, 0] <= bottom)
                    ).mean()
                    if enclosed < overlap:
                        lower_scale = scale
                    else:
                        upper_scale = scale
                scale = upper_scale
                box = np.array([
                    center_x - width * scale / 2,
                    center_y - height * scale / 2,
                    center_x + width * scale / 2,
                    center_y + height * scale / 2,
                ], dtype=np.float32)
            prompt_data['box'] = box

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'image_meta_dict':image_meta_dict,
            **prompt_data,
        }