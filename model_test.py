# %% [markdown]
# # import

# %%
import torch
import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch.nn.functional as F
import random
from typing import OrderedDict
import os
import pandas as pd
import numpy as np
from vilt.transforms import pixelbert_transform
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import defaultdict
import wandb

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")


# %% [markdown]
# # config

# %%
class config:
    debug = True
    sensor_only = False
    crop_name = "total"
    class_num = 4
    
    exp_name = "ViST"
    seed = 101
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    train_batch_size = 32
    valid_batch_size = 4
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    n_fold = 5

    model_name = "sensorViLOnlyTransformerSS" #仅图片
    # model_name = "sensorOnlyViLTransformerSS"  #仅vilt传感器
    # model_name = "sensorViLTransformerSS"  #vilt图像+传感器
    # model_name = "DNNF1"  #DNNF1图像+传感器
    # model_name = "DNNF1PictureOnly"  #DNNF1图像
    # model_name = "DNNF1SensorOnly"  #DNNF1传感器

    # model_name = "DNNF2"  #DNNF2图像+传感器
    # model_name = "DNNF2PictureOnly"  #DNNF1图像
    # model_name = "DNNF2SensorOnly"  #DNNF1传感器
    # wandb 
    # wandb_name = "vilt|大豆|290图像加传感器"
    # wandb_name = "vilt|大豆|290仅传感器"
    wandb_name = "vilt|大豆|290仅图片"

    # wandb_name = "DNNF1|大豆|290图像加传感器"
    # wandb_name = "DNNF1|大豆|290仅图像"
    # wandb_name = "DNNF1|大豆|290仅传感器"
    
    # wandb_name = "DNNF2|大豆|290图像加传感器"
    # wandb_name = "DNNF2|大豆|290仅图像"
    # wandb_name = "DNNF2|大豆|290仅传感器"
    

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    img_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Sensor
    # senser_input_num = 11 # 翔冠的传感器参数
    senser_input_num = 19 # 天航的传感器参数
    
    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522 # vocabulary词汇数量
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768  # 嵌入向量大小
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-3 #0.0015#2e-3 #
    weight_decay = 1e-4 # 0.01 ->1e-4
    decay_power = 1
    max_epoch = 50
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    # T_max = 8000/train_batch_size*max_epoch 
    # T_max = 4632/train_batch_size*max_epoch # total 7237.5
    T_max = 2126/train_batch_size*max_epoch # soybean 3321.875

    # Downstream Setting
    get_recall_metric = False


    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = "weights/vilt_200k_mlm_itm.ckpt"
    # load_path = "save_model_dict.pt"
    num_workers = 1
    precision = 16

    # CBP 算法1,random maclaurin Projection参数
    RMP_d = 10000



if config.debug:
    config.max_epoch = 5
print("当前device=",config.device)

# %%
def setup_seed(seed):

    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    #os.environ['PYTHONHASHSEED'] = str(seed)
    
setup_seed(config.seed)

# %% [markdown]
# ## model build

# %%
from models.SemanticEstimation import SemanticEstimation
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import models

from models.CNNTransformer import CNNTransformer
from models.RiceFusion import RiceFusion
from models.RiceTransformer import RiceTransformer
from models.ViST import *
from models.resnet import *
from models.vilt_ import *
from models.DNNF1 import *
from models.DNNF2 import *
from models.RiceFusionMLP import *
from models.RiceFusionCNN import *
from models.BilinearPooling import *
from models.CompactBilinearPoolingRMP import *
from models.CompactBilinearPoolingTSP import *

def build_model(model_name: str,pre_train):
    if model_name[:6] == "resnet50":
        model = pretrainedmodels.__dict__[config.model_name](
            num_classes=1000, pretrained='imagenet')
        dim_feats = model.last_linear.in_features  # =2048
        nb_classes = 1
        model.last_linear = nn.Linear(dim_feats, nb_classes)
        return model
    if model_name == "se_resnet50":
        model = pretrainedmodels.__dict__[config.model_name](
            num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(204800, 1,bias=True)
        return model
    if model_name == "efficientnet-b4": # efficient net
        # refer:https://github.com/lukemelas/EfficientNet-PyTorch#example-classification
        nb_classes = 1
        if pre_train:
            model = EfficientNet.from_pretrained(config.model_name)# 'efficientnet-b4'
        else:
            model = EfficientNet.from_name(config.model_name)# 'efficientnet-b4'
        model._fc = nn.Linear(1792, nb_classes)
        return model
        
    if model_name == "ViST":
        model = ViST(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
    if model_name == "ViST2":
        model = ViST2(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
    if model_name == "sensorViST":
        model = sensorViST(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
    if model_name == "imageViST":
        model = imageViST(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
        
    if model_name == "sensorOnlyViLTransformerSS": #仅传感器
        model = sensorOnlyViLTransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
    if model_name == "sensorViLOnlyTransformerSS": # 仅vit图像
        model = sensorViLOnlyTransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
        
    if model_name == "sensorResnet50TransformerSS":
        model = sensorResnet50TransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model
    if model_name == "sensorResnet101TransformerSS":
        model = sensorResnet101TransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model

    if model_name == "sensorViLTransformerSS":
        model = sensorViLTransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1,config=config)
        return model

    if model_name == "DNNF1":
        model = DNNF1(sensor_nums=config.senser_input_num,config=config)
        return model
    if model_name == "DNNF1PictureOnly":
        model = DNNF1PictureOnly(sensor_nums=config.senser_input_num,config=config)
        return model
    if model_name == "DNNF1SensorOnly":
        model = DNNF1SensorOnly(sensor_nums=config.senser_input_num,config=config)
        return model
        
    if model_name == "DNNF2":
        model = DNNF2(sensor_nums=config.senser_input_num,config=config)
        return model
    if model_name == "DNNF2PictureOnly":
        model = DNNF2PictureOnly(sensor_nums=config.senser_input_num,config=config)
        return model
    if model_name == "DNNF2SensorOnly":
        model = DNNF2SensorOnly(sensor_nums=config.senser_input_num,config=config)
        return model
    # RiceFusion对比模型
    if model_name == "RiceFusionMLP":
        model = RiceFusionMLP(sensor_nums=config.senser_input_num,config=config)
        return model

    if model_name == "RiceFusionCNN":
        model = RiceFusionCNN(config=config)
        return model
    
    if model_name == "RiceFusion":
        return RiceFusion(sensor_nums=config.senser_input_num,config=config)
    if model_name == "RiceTransformer":
        return RiceTransformer(sensor_nums=config.senser_input_num,config=config)
    if model_name == "CNNTransformer":
        return CNNTransformer(sensor_nums=config.senser_input_num,config=config)
    
    if model_name == "BilinearPooling":
        return BilinearPooling(sensor_nums=config.senser_input_num,config=config)
    
    if model_name == "CompactBilinearPoolingRMP":
        return CompactBilinearPoolingRMP(sensor_nums=config.senser_input_num,config=config)
    if model_name == "CompactBilinearPoolingTSP":
        return CompactBilinearPoolingTSP(sensor_nums=config.senser_input_num,config=config)
    if model_name == "SemanticEstimation":
        return SemanticEstimation(sensor_class_n=config.senser_input_num,output_class_n = config.class_num,config=config)
    raise Exception("模型未定义")
    

# %% [markdown]
# # Test

# %%

model = build_model("SemanticEstimation",True).to(config.device)
size = (2, 145, 768)
image = torch.randn((32,3,384,384)).to(config.device)
sensor_input = torch.randn((32,1,config.senser_input_num)).to(config.device)
batch = {"image":image,"sensor":sensor_input}

output = model(batch)
print(output.shape)


