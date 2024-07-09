import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor
from torchvision.transforms import ToPILImage, ToTensor, Compose


class LLdataset:
    def __init__(self, config):
        self.config = config
        self.train_dataset = AllWeatherDataset(
            os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'train'),
            patch_size=self.config.data.patch_size,
            filelist='{}_train_{}.txt'.format(self.config.data.train_dataset, self.config.data.data_volume))
        self.val_dataset = AllWeatherDataset(
            os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'val'),
            patch_size=self.config.data.patch_size,
            filelist='{}_val_{}.txt'.format(self.config.data.val_dataset, self.config.data.data_volume), train=False)

    def server_get_loaders(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def get_evaluation_loaders(self):
        eval_dataset = EvaluationDataset(self.config.data.eval_dir)
        val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return val_loader

    def fl_get_loaders(self):
        train_loader = self.iid_partition_loader(self.train_dataset,
                                                 batch_size=self.config.training.fl_batch_size,
                                                 n_clients=self.config.training.fl_clients)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

    def iid_partition_loader(self, dataset, batch_size, n_clients):
        """
        partition the dataset into a dataloader for each client, iid style
        """
        m = len(dataset)
        assert m % n_clients == 0
        m_per_client = m // n_clients
        assert m_per_client % batch_size == 0

        client_data = torch.utils.data.random_split(
            dataset,
            [m_per_client for x in range(n_clients)]
        )
        client_loader = [
            torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, pin_memory=True)
            for x in client_data
        ]
        return client_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('low', 'high') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        gt_name = self.gt_names[index].replace('\n', '')
        img_id = re.split("\\\\", input_name)[-1][:-4]
        #  if self.dir else PIL.Image.open(input_name)  if self.dir else PIL.Image.open(gt_name)
        input_img = Image.open(self.dir + input_name)
        gt_img = Image.open(self.dir + gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)
        input_img = input_img[:3, ...]

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()

        self.dir = dir
        input_names = list()
        for file_name in os.listdir(self.dir):
            input_names.append(os.path.join(self.dir, file_name))
        self.input_names = input_names
        self.transforms = Compose([
            ToTensor()
        ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        img_id = re.split("\\\\", input_name)[-1][:-4]
        #  if self.dir else PIL.Image.open(input_name)  if self.dir else PIL.Image.open(gt_name)
        input_img = Image.open(input_name)

        input_img = self.transforms(input_img)
        input_img = input_img[:3, ...]

        return input_img, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
