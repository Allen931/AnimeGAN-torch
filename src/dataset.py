import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import normalize_input, compute_data_mean
from utils.edge_smooth import edge_smooth
from torchvision.transforms import RandomResizedCrop

extension = {'.jpg', '.png', '.bmp','.jpeg', '.JPG', '.JPEG'}

class AnimeDataSet(Dataset):
    def __init__(self, args, transform=None):
        """
        folder structure:
            - {data_dir}
                - photo
                    1.jpg, ..., n.jpg
                - {dataset}  # E.g Hayao
                    1.jpg, ..., n.jpg
        """
        train_dir = os.path.join(args.train_dir)
        dataset = args.dataset
        anime_dir = os.path.join(args.anime_dir)

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f'Folder {train_dir} does not exist')

        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')

        self.mean = compute_data_mean(anime_dir)
        print(f'Mean(B, G, R) of {dataset} are {self.mean}')

        self.debug_samples = args.debug_samples or 0
        self.image_files = {}
        self.train = 'train_photo'
        self.anime = dataset
        self.dummy = torch.zeros(3, 256, 256)

        # for opt in [self.train, self.anime]:
        #     folder = os.path.join(data_dir, opt)
        #     files = os.listdir(folder)

        self.image_files[self.anime] = [os.path.join(anime_dir, fi) for fi in os.listdir(anime_dir)]
        self.image_files[self.train] = []
        for root, subdirs, files in os.walk(train_dir):
            for file in files:
                if os.path.splitext(file)[-1] in extension:
                    self.image_files[self.train].append(os.path.join(root, file))

        self.transform = transform

        print(f'Dataset: real {len(self.image_files[self.train])} anime {self.len_anime}')

    def __len__(self):
        return max(len(self.image_files[self.train]), len(self.image_files[self.anime]))

    @property
    def len_anime(self):
        return len(self.image_files[self.anime])

    def __getitem__(self, index):
        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray, anime_smooth = self.load_anime(anm_idx)

        return image, anime, anime_gray, anime_smooth

    def load_photo(self, index):
        fpath = self.image_files[self.train][index]
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._transform(image, addmean=False)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def load_anime(self, index):
        fpath = self.image_files[self.anime][index]
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = RandomResizedCrop(
            256, scale=(0.4, 1.0), ratio=(0.85, 1.15)
        )(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))
        image = image.squeeze(0).permute(1, 2, 0).numpy()

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = np.transpose(image_gray, (2, 0, 1))

        image_smooth = self.load_anime_smooth(image)

        image = self._transform(image, addmean=True)
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(image_gray), torch.from_numpy(image_smooth)

    def load_anime_smooth(self, image):
        image = edge_smooth(image)
        image = self._transform(image, addmean=False)
        image = np.transpose(image, (2, 0, 1))
        return image

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean

        return normalize_input(img)


class ValidationSet(Dataset):
    def __init__(self, args, mean, transform=None):
        self.mean = mean
        self.transform = transform

        val_dir = os.path.join(args.val_dir)
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f'Folder {val_dir} does not exist')

        files = os.listdir(val_dir)
        self.image_files = [os.path.join(val_dir, fi) for fi in files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        fpath = self.image_files[index]
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._transform(image, addmean=False)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean

        return normalize_input(img)


class TestSet(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        test_dir = os.path.join(args.test_dir)
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f'Folder {test_dir} does not exist')

        files = os.listdir(test_dir)
        self.image_files = [os.path.join(test_dir, fi) for fi in files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        fpath = self.image_files[index]
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._transform(image)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def _transform(self, img):
        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32)

        return normalize_input(img)