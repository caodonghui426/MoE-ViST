# import gradio as gr
import torch
import copy
import time
import requests
import io
import numpy as np
import re
import cv2
import ipdb
import os
from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.modules.vilt_regress import regressViLTransformerSS
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer

from torchsummary import summary

@ex.automain
def main(_config):
    # pl.seed_everything(0)
    _config = copy.deepcopy(_config)
    print("\033[1;32m config: \033[0m",_config)
    model = regressViLTransformerSS(_config,sensor_class_n= 12)
    # print(model)
    # torch.save(model.state_dict(), 'embedding_test_dict.pt')


    # print(model)
    model.setup("test")
    model.eval()
    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)
    def infer(img_filename, sensor):
        try:
            img_path = os.path.join('pictures',img_filename)
            image = Image.open(img_path).convert("RGB")
            img = pixelbert_transform(size=384)(image) # 将图像数据归一化torch.Size([3, 384, 576])
            img = torch.tensor(img)
            img = torch.unsqueeze(img, 0) # torch.Size([1, 3, 384, 576])
            img = img.to(device)
            print("img.shape:",img.shape)
        except :
            print("图片加载失败！")
            raise

        batch = {"text": [""], "image": [None]}
        batch["image"][0] = img

        batch['sensor_masks'] = torch.ones(1,1).to(device)
        with torch.no_grad():
            batch['sensor'] = sensor.to(device)       
            infer = model(batch)

            # print(infer)
            sensor_emb, img_emb = infer["sensor_feats"], infer["image_feats"]# torch.Size([1, 23, 768]) torch.Size([1, 217, 768])
            cls_output = infer['cls_output']
         
    
        return [cls_output]

    examples=[
            [
                "6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the [MASK] [MASK] in front of [MASK] on a [MASK] [MASK].",
                0,
            ],
            [
                "6212487_1cca7f3f_1024x1024.jpg",
                "a a a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day",
                4,
            ],
            [
                "6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                11,
            ],
            [
                "6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                15,
            ],
            [
                "6212487_1cca7f3f_1024x1024.jpg",
                "a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day.",
                18,
            ],
            [
                "800px-Living_Room.jpg",
                "a room with a [MASK], a [MASK], a [MASK], and a [MASK].",
                0,
            ],
            [
                "800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                5,
            ],
            [
                "800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                8,
            ],
            [
                "800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                11,
            ],
            [
                "800px-Living_Room.jpg",
                "a room with a rug, a chair, a painting, and a plant.",
                15,
            ],
        ],

    n = 1
    sensor = torch.randn(1,1,12)
    out = infer(examples[0][n][0],sensor)
    # print("out:",out,"000\n")
    # print("out0.shape:",out[0].shape)
    # cv2.imwrite('output.png',out[0])
    return  out
