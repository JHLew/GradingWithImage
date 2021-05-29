import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
import os
from glob import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from random import randint
from torchvision.transforms import RandomAffine


class Data(Dataset):
    def __init__(self, path, target_size=(512, 32), is_train=True):
        paths = glob(path)
        self.jsons = []
        self.dirs = []
        for p in paths:
            if '.zip' in p:
                continue
            elif '.json' in p:
                self.jsons.append(p)
            else:
                self.dirs.append(p)

        self.hand_list = []
        self.print_list = []
        for d in self.dirs:
            img_list = glob(os.path.join(d, '*', '*'))
            if 'hand' in d:
                self.hand_list += img_list
            elif 'print' in d:
                self.print_list += img_list

        self.val_hand = self.hand_list[-100:]
        self.val_print = self.print_list[-100:]
        self.hand_list = self.hand_list[:-100]
        self.print_list = self.print_list[:-100]

        self.n_hand = len(self.hand_list)
        self.n_print = len(self.print_list)

        self.size = target_size
        self.aug_affine = RandomAffine(degrees=2, shear=(-2, 2, -2, 2), fillcolor=255)
        self.is_train = is_train

    def __len__(self):
        return self.n_hand + self.n_print

    def __getitem__(self, idx):
        if idx >= self.n_hand:  # print
            idx = idx - self.n_hand
            img = self.print_list[idx]
            label = 1
        else:
            img = self.hand_list[idx]
            label = 0

        img = Image.open(img).convert('L')

        new_h = randint(self.size[1] // 2, self.size[1])
        w, h = img.size
        scale = h / new_h
        img = img.resize((int(w // scale), new_h), resample=Image.BICUBIC)

        background_img = Image.new('L', self.size, 255)

        new_w, _ = img.size  # new w, h
        h_offset = self.size[1] - new_h
        h_offset = randint(0, h_offset)

        if new_w > self.size[0]:
            diff = new_w - self.size[0]
            w_offset = randint(0, diff)
            img = img.crop((w_offset, 0, w_offset + self.size[0], new_h))
            w_offset = 0
        else:
            w_offset = self.size[0] - new_w
            w_offset = randint(0, w_offset)

        background_img.paste(img, (w_offset, h_offset))

        # affine transform augmentation
        if self.is_train:
            background_img = self.aug_affine(background_img)

        flag = randint(0, 2)
        if flag == 2:  # 1/3 prob. mixed.
            flag = randint(0, 1)
            if flag == 0:  # same
                if label == 1:  # main img is printed
                    sub_i = randint(0, self.n_print - 1)
                    sub_img = self.print_list[sub_i]
                else:  # main img is handwritten
                    sub_i = randint(0, self.n_hand - 1)
                    sub_img = self.hand_list[sub_i]

            else:  # different
                if label == 0:  # main img is handwritten
                    sub_i = randint(0, self.n_print - 1)
                    sub_img = self.print_list[sub_i]
                else:  # main img is printed
                    sub_i = randint(0, self.n_hand - 1)
                    sub_img = self.hand_list[sub_i]
                label = 2

            sub_img = Image.open(sub_img).convert('L')

            sub_new_h = randint(self.size[1] // 2, self.size[1])
            sub_w, sub_h = sub_img.size
            sub_scale = sub_h / sub_new_h
            sub_img = sub_img.resize((int(sub_w // sub_scale), sub_new_h), resample=Image.BICUBIC)

            sub_new_w, _ = sub_img.size
            sub_h_offset = self.size[1] - sub_new_h
            sub_h_offset = randint(0, sub_h_offset)

            if sub_new_w > self.size[0] // 2:
                sub_crop_size = randint(40, self.size[0] // 2)
                sub_w_offset = randint(0, sub_new_w - sub_crop_size)
                sub_img = sub_img.crop((sub_w_offset, 0, sub_w_offset + sub_crop_size, sub_new_h))
                sub_new_w = sub_crop_size

            sub_w_offset = self.size[0] - sub_new_w
            sub_w_offset = randint(0, sub_w_offset)

            white = Image.new('L', (sub_img.size[0], self.size[1]), color=255)
            background_img.paste(white, (sub_w_offset, 0))
            background_img.paste(sub_img, (sub_w_offset, sub_h_offset))

        img = background_img

        if self.is_train:
            # blur augmentation
            img = np.array(img)
            k = randint(1, 2)
            if not k == 0:
                k = 2 * k - 1
                img = cv2.GaussianBlur(img, ksize=(k, k), sigmaX=0)

            img = to_tensor(img)
            label = torch.LongTensor([label])

            # random noise
            noise_level = float(randint(0, 750)) / 10000
            if not noise_level == 0:
                noise = torch.randn_like(img) * noise_level
                img = img + noise
        else:
            img = to_tensor(img)
            label = torch.LongTensor([label])

        img = torch.clamp(img, 0, 1)

        return img, label


train_dataset = Data('./dataset/*')
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)

test_dataset = Data('./dataset/*', is_train=False)
test_dataset.hand_list = train_dataset.val_hand
test_dataset.print_list = train_dataset.val_print
test_dataset.n_hand = len(test_dataset.hand_list)
test_dataset.n_print = len(test_dataset.print_list)

test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=2)
