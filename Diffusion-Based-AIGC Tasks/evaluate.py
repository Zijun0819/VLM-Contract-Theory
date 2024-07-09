import argparse
import csv
import os
import re
import time

# import lpips
import numpy as np
import torch
import yaml
from PIL import Image
# from pytorch_msssim import ssim
from torchvision.transforms import ToTensor

import datasets
from models import DenoisingDiffusion, DiffusiveRestoration

# lpips_model = lpips.LPIPS(net='alex')


def parse_args_and_config():
    with open(os.path.join("configs", "LOLv1.yml"), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default=f'ckpt\\{new_config.data.data_volume}_{new_config.model.model_size}model_latest.pth', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results\\test', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    parser.add_argument('--label_dir', default="data\\Image_restoration\\LL_dataset\\Construction\\val\\high_L",
                        type=str,
                        help="Location of image under normal light condition")
    args = parser.parse_args()

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}_{}'".format(config.data.val_dataset, config.data.data_volume))
    DATASET = datasets.__dict__[config.data.type](config)
    val_loader = DATASET.get_evaluation_loaders()

    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    time_start = time.time()
    model.restore(val_loader)
    print("==>Total time elapsed: {:.4f}".format(time.time() - time_start))


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = ToTensor()(image).unsqueeze(0)  # 转换为torch tensor并添加batch维度
    return image


# def cal_lpips_ssim(img1_path: str, img2_path: str) -> tuple:
#     img1 = load_image(img1_path)
#     img2 = load_image(img2_path)
#
#     img_id = re.split("\\\\", img1_path)[-1][:-4]
#     # 计算 LPIPS 距离
#     lpips_distance = lpips_model(img1, img2)
#     # calculate SSIM
#     ssim_distance = ssim(img1, img2, data_range=1.0)
#     print(f'Image ID: {img_id}, LPIPS distance: {lpips_distance.item()}, SSIM distance: {ssim_distance.item()}')
#
#     return img_id, 1-lpips_distance.item(), ssim_distance.item()


# def get_metrics():
#     args, config = parse_args_and_config()
#
#     eval_dir = os.path.join(args.image_folder, f"{config.data.val_dataset}_{config.data.data_volume}")
#     print(f"Evaluate the image generated at path: {eval_dir}")
#     score_save_path = os.path.join(args.image_folder, f"score_{config.data.data_volume}.csv")
#     metrics_list = list()
#
#     for file_name in os.listdir(eval_dir):
#         img_id, lpips_, ssim_ = cal_lpips_ssim(os.path.join(eval_dir, file_name), os.path.join(args.label_dir, file_name))
#         metrics_list.append((img_id, lpips_+ssim_))
#
#     with open(score_save_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         [writer.writerow(x) for x in metrics_list]

if __name__ == '__main__':
    main()
    # get_metrics()
