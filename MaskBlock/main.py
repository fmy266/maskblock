import torch
import torch.nn as nn
import torchvision
import timm
from math import ceil
import sys
sys.path.append("..")
from black_box_attack import attack_method


def drop_image(drop_rate, device):
    def f(image):
        return image * torch.rand_like(image, device=device) >= drop_rate
    return f


def drop_area(area_size = 2, image_size = 224, batch_size = 16, device = torch.device("cpu")):
    mask_list = []
    mask_area_num = ceil(image_size / area_size)
    for i in range(mask_area_num):
        for j in range(mask_area_num):
            mask = torch.ones(3, image_size, image_size, device = device)
            mask[:, i * area_size: (i+1) * area_size, j * area_size: (j+1) * area_size] = 0.
            mask.unsqueeze_(dim = 0)
            mask = mask.repeat(batch_size, 1, 1, 1)
            mask_list.append(mask)     
    mask = torch.cat(mask_list, dim = 0)
    
    def f(image):
        return image.repeat(mask_area_num ** 2, 1, 1, 1) * mask
    return f



if __name__ == "__main__":

    batch_size = 128
    iter_num = 10
    step = 0.32 / iter_num * 2
    area_size = 56
    data_path = "./data" # you need to specify the data path
    device = torch.device("cuda:0")
    imagenet_data = torchvision.datasets.ImageNet(data_path)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=batch_size,
                                              shuffle=True)
    # you can select your model (refer to pytorch pretrained models)
    proxy_model = torchvision.models.densenet121(pretrained = True).to(device).eval()
    black_model = torchvision.models.inception_v3(pretrained = True).to(device)
    drop_func = drop_area(area_size = area_size, image_size = 224, batch_size = batch_size, device = device)

    attack = attack_method.BaseAttack(epsilon=0.32, iter_num=iter_num, step=step, device = device,
                                                                data_preprocessing = drop_func, distortion_preprocessing = drop_func,
                                                                label_preprocessing = lambda x: x.repeat(ceil(224/area_size)**2))
    asr = robust_attack.attack(black_model, proxy_model, nn.CrossEntropyLoss(), data_loader)
    print(asr)