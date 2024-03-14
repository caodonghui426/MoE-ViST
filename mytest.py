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
    T_max = 12355*4/train_batch_size*max_epoch # soybean 3321.875

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
    config.max_epoch = 2
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
# # wandb

# %%
if config.debug == True:
    os.environ["WANDB_MODE"] = 'dryrun' # 离线模式
try:
    # wandb.log(key="*******") # if debug
    wandb.login() # storage in ~/.netrc file
    anonymous = None
except:
    anonymous = "must"
    print('\nGet your W&B access token from here: https://wandb.ai/authorize\n')


os.environ["WANDB_MODE"] = 'dryrun' # 离线模式


# %% [markdown]
# # 数据

# %%
def fetch_df(crop_name):
    if crop_name == "soybean":
        df_tianhang = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-soybean.csv")
        df_tianhang['image_path'] = df_tianhang['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_soybean',x.split('/')[-1]))
    elif crop_name == "rice":
        df_tianhang = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-rice.csv")
        df_tianhang['image_path'] = df_tianhang['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_rice',x.split('/')[-1]))
    elif crop_name == "corn":
        
        df_tianhang = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-corn.csv")
        df_tianhang['image_path'] = df_tianhang['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_corn',x.split('/')[-1]))
    elif crop_name == "total":
        df_tianhang_soybean = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-soybean.csv")
        df_tianhang_soybean['image_path'] = df_tianhang_soybean['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_soybean',x.split('/')[-1]))

        df_tianhang_rice = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-rice.csv")
        df_tianhang_rice['image_path'] = df_tianhang_rice['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_rice',x.split('/')[-1]))

        df_tianhang_corn = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-corn.csv")
        df_tianhang_corn['image_path'] = df_tianhang_corn['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_corn',x.split('/')[-1]))

        df_tianhang = pd.concat([df_tianhang_soybean,df_tianhang_rice,df_tianhang_corn])
    df_tianhang['label'] = df_tianhang['LAI']
    df_tianhang = df_tianhang.dropna()
    df_tianhang = df_tianhang.reset_index()
    # print(df_tianhang.shape)
    number_title = []
    # 归一化数值列
    recorder = {}
    for title in df_tianhang:
        # print(df_xiangguan[title].head())
        if title == 'raw_label':
            continue
        if df_tianhang[title].dtype != "object":
            
            number_title.append(title)
            x_min = df_tianhang[title].min()
            x_max = df_tianhang[title].max()
            # print(x_min,x_max)
            recorder[title] = (x_min,x_max)
            df_tianhang[title] = df_tianhang[title].map(lambda x:(x-x_min + 0.01)/(x_max - x_min))
    # print(number_title)
    # print(recorder)

    # 选择传感器列
    if crop_name=="corn":
        config.senser_input_num = 17
        tianhang_sensor = ['co2', 'stemp', 'stemp2', 'stemp3', 'stemp5', 'shumi', 'shumi2', 'shumi3', 'shumi5', 'humi', 'pm10', 'pm25', 'press', 'solar', 'temp', 'wind_d', 'wind_sp']
    else:
        config.senser_input_num = 19
        tianhang_sensor = ['co2', 'stemp', 'stemp2', 'stemp3', 'stemp4', 'stemp5', 'shumi', 'shumi2', 'shumi3', 'shumi4', 'shumi5', 'humi', 'pm10', 'pm25', 'press', 'solar', 'temp', 'wind_d', 'wind_sp']
    if crop_name == "total":
        config.senser_input_num = 17
        tianhang_sensor = ['co2', 'stemp', 'stemp2', 'stemp3', 'stemp5', 'shumi', 'shumi2', 'shumi3', 'shumi5', 'humi', 'pm10', 'pm25', 'press', 'solar', 'temp', 'wind_d', 'wind_sp']
    df_tianhang['sensor'] = df_tianhang[tianhang_sensor].values.tolist()
    # print("input dim:",len(tianhang_sensor))
    
    # 筛选仅传感器信息
    if config.sensor_only:
    # del df_tianhang['pic_key']
        df_tianhang.drop_duplicates(subset=['pic_key'],inplace=True,ignore_index=True)
    # print("*********************df shape:",df_tianhang.shape)
    
    # debug 特判
    df=df_tianhang
    if config.debug:
        df = df[:100]
    return df

# fetch_df(config.crop_name)

# %%
fetch_df('soybean').shape

# %% [markdown]
# create folds

# %%
def creat_folds(df):
    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)  
    for fold, (train_idx, val_idx) in enumerate(skf.split(df,df.date)):
        df.loc[val_idx, 'fold'] = fold
    print(df.groupby(['fold'])['label'].count())   
    return df 


# %% [markdown]
# # dataset
# 

# %%
myTransforms = transforms.Compose([
    transforms.Resize((config.img_size,config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.4870, 0.5287, 0.4776],
    std=[0.1639, 0.1735, 0.1617],
)
])

def load_img(path):
    img =  Image.open(path).convert('RGB')
    img = myTransforms(img)
    return img

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.sensors = df['sensor'].tolist()
        self.img_paths  = df['image_path'].tolist()   
        if self.label:
            self.labels = df['label'].tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = load_img(img_path)
        sensor = self.sensors[index]
        sensor = torch.tensor(sensor).unsqueeze(0) #[1,n]
        if self.label:
            label = self.labels[index]
            return torch.tensor(img).to(torch.float), torch.tensor(sensor).to(torch.float),torch.tensor(label).to(torch.float)
         

        else:
            return torch.tensor(img).to(torch.float), torch.tensor(sensor).to(torch.float)


# %% [markdown]
# # dataloader

# %%
def fetch_dataloader(fold:int,df):
    train_df = df.query("fold!=@fold").reset_index(drop=True)

    valid_df = df.query("fold==@fold").reset_index(drop=True)
    print("train_df.shape:",train_df.shape)
    print("valid_df.shape:",valid_df.shape)

    train_data  = BuildDataset(df=train_df,label=True)
    valid_data = BuildDataset(df=valid_df,label=True)

    train_loader = DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.valid_batch_size,shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=config.test_batch_size,shuffle=False)
    return train_loader,valid_loader

def fetch_dataloader_ubiquatous():
    train_df = pd.concat((fetch_df('soybean'),fetch_df('rice')),axis=0,join='inner').reset_index(drop=True)

    valid_df = fetch_df('corn').reset_index(drop=True)
    print("train_df.shape:",train_df.shape)
    print("valid_df.shape:",valid_df.shape)

    train_data  = BuildDataset(df=train_df,label=True)
    valid_data = BuildDataset(df=valid_df,label=True)

    train_loader = DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.valid_batch_size,shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=config.test_batch_size,shuffle=False)
    return train_loader,valid_loader
def fetch_dataloader_ubiquatous_single_crop_test(fold:int,df,crop_name):#以一种作物为测试
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    if crop_name == "soybean":
        valid_df = pd.read_csv("/home/junsheng/ViLT/data/ubiquitous_soybean.csv")
    elif crop_name == "corn":
        valid_df = pd.read_csv("/home/junsheng/ViLT/data/ubiquitous_corn.csv")
    elif crop_name == "rice":
        valid_df = pd.read_csv("/home/junsheng/ViLT/data/ubiquitous_rice.csv")
    print("train_df.shape:",train_df.shape)
    print("valid_df.shape:",valid_df.shape)

    train_data  = BuildDataset(df=train_df,label=True)
    valid_data = BuildDataset(df=valid_df,label=True)

    train_loader = DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.valid_batch_size,shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=config.test_batch_size,shuffle=False)
    return train_loader,valid_loader

# %% [markdown]
# 计算图像均值标准差

# %%
def get_mean_std_value(loader):
    '''
    求数据集的均值和标准差
    :param loader:
    :return:
    '''
    data_sum,data_squared_sum,num_batches = 0,0,0
       
    pbar = tqdm(enumerate(loader), total=len(loader), desc='caculating ')    
    # for data,sensor,label  in loader:
    for step,(data,sensor,label)  in pbar:
        # data: [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        # data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
        data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,height,width,channels]
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        # data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,height,width,channels]
        # 统计batch的数量
        num_batches += 1

       
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std
df = fetch_df(config.crop_name)
df = creat_folds(df)
# train_loader,_ = fetch_dataloader(fold=0,df=df)
# mean,std = get_mean_std_value(train_loader)
# print('mean = {},std = {}'.format(mean,std))

# %% [markdown]
# # model

# %% [markdown]
# ## model build

# %%
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
from models.SemanticEstimation import SemanticEstimation


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
        return SemanticEstimation(sensor_class_n=config.senser_input_num,output_class_n = 1,config=config)
    
    raise Exception("模型未定义")
    

# %% [markdown]
# # 损失函数

# %%
criterion = F.mse_loss #均方误差损失函数
criterion_mae = nn.L1Loss()
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
def MAPE(y_true,y_pred):
    """计算MAPE误差，除数如果为0或者太小，则返回数值会很大

    Args:
        y_true (_type_): ground truth
        y_pred (_type_): 预测值

    Returns:
        _type_: _description_
    """
    return mean_absolute_percentage_error(y_true,y_pred)

def SMAPE(y_true, y_pred):
    """计算smape

    Args:
        y_true (torch.tensor): 真实值
        y_pred (torch.tensor): 预测值

    Returns:
        tensor: 一个数，如返回50，则表示50%
    """
    return 2.0 * torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true))) * 100.0


# %% [markdown]
# # train one epoch

# %%



def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (img, sensor,label) in pbar:         
        # img = img.to(device, dtype=torch.float)
        # sensor  = sensor.to(device, dtype=torch.float)
        # label  = label.to(device, dtype=torch.float)
        batch_size = img.size(0)
        
        batch = {"image":img,"sensor":sensor}

        y_pred = model(batch)
        label = label.to(config.device).unsqueeze(1)
        loss = criterion(y_pred['cls_output'], label)
        
        #一坨优化
        optimizer.zero_grad()#每一次反向传播之前都要归零梯度
        loss.backward()      #反向传播
        optimizer.step()     #固定写法
        scheduler.step()
     
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')

    
        
        
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

# %% [markdown]
# # valid one epoch

# %%
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, optimizer):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    running_loss_mae = 0.0
    running_loss_smape = 0.0
    running_loss_mape = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (img, sensor,label) in pbar:               
        
        
        batch_size = img.size(0)
        batch = {"image":img,"sensor":sensor}

        y_pred  = model(batch)
        label = label.to(config.device).unsqueeze(1)

        loss = criterion(y_pred['cls_output'], label)
        loss_mae = criterion_mae(y_pred['cls_output'], label)
        loss_smape = SMAPE(label,y_pred['cls_output'])
        loss_mape = MAPE(label.cpu(),y_pred['cls_output'].cpu())
        
        running_loss += (loss.item() * batch_size)
        running_loss_mae += (loss_mae.item() * batch_size)
        running_loss_smape += (loss_smape.item() * batch_size)
        running_loss_mape += (loss_mape.item() * batch_size)

        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_loss_mae = running_loss_mae / dataset_size
        epoch_loss_smape = running_loss_smape / dataset_size
        epoch_loss_mape = running_loss_mape / dataset_size
        
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss_mse=f'{epoch_loss:0.4f}',
        valid_loss_mae=f'{epoch_loss_mae:0.4f}',
        valid_loss_smape=f'{epoch_loss_smape:0.4f}',
        valid_loss_mape=f'{epoch_loss_mape:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss,epoch_loss_mae,epoch_loss_smape,epoch_loss_mape#MSE，MAE,r2 score,mape

# %% [markdown]
# # train

# %%

def run_training(model, optimizer, scheduler, device, num_epochs,train_loader,valid_loader):
     # init wandb
    run = wandb.init(project=config.exp_name,
                    config={k: v for k, v in dict(vars(config)).items() if '__' not in k},
                    # config={k: v for k, v in dict(config).items() if '__' not in k},
                    anonymous=anonymous,
                    # name=f"vilt|fold-{config.valid_fold}",
                    name=config.wandb_name,
                    # group=config.wandb_group,
                    )
    wandb.watch(model, log_freq=100)

    best_loss = 9999
    best_valid_loss = 9999
    history = defaultdict(list)
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)
        val_loss,val_loss_mae,val_loss_smape,val_loss_mape = valid_one_epoch(model,valid_loader,device=device,optimizer=optimizer) # epoch_loss,epoch_loss_mae,epoch_loss_smape,epoch_loss_mape
        history['Train Loss'].append(train_loss)
        history['Valid Loss MSE'].append(val_loss)
        history['Valid Loss MAE'].append(val_loss_mae)
        history['Valid Loss SMAPE'].append(val_loss_smape)
        history['Valid Loss MAPE'].append(val_loss_mape)

        wandb.log({"Train Loss": train_loss,
                    "Valid Loss": val_loss,
                    "Valid Loss MAE": val_loss_mae,
                    "Valid Loss SMAPE": val_loss_smape,
                    "Valid Loss MAPE": val_loss_mape,
                "lr": scheduler.get_last_lr()[0]
                })
        if best_valid_loss >= val_loss:
            best_valid_loss = val_loss
            model_file_path = os.path.join(wandb.run.dir,"epoch-{}-{}.bin".format(epoch,wandb.run.id))
            # model_file_path = os.path.join(wandb.run.dir,"epoch-best.bin")
            run.summary["Best Epoch"] = epoch
            torch.save(model, model_file_path)
            print("model save to", model_file_path)
               
    os.system("cp /home/junsheng/ViLT/my_vilt_total.ipynb {}".format(wandb.run.dir))
    run.finish()
    return model, history

# %% [markdown]
# run train

# %%
def run(crop_name:str,model_name:str,wandb_name:str,sensor_only:bool):
    config.model_name = model_name
    config.wandb_name = wandb_name
    config.sensor_only = sensor_only

    df = fetch_df(crop_name)
    df = creat_folds(df)

    train_loader,valid_loader = fetch_dataloader(fold=0,df=df)
    # train_loader,valid_loader = fetch_dataloader_ubiquatous()


    model = build_model(config.model_name,True)
    model.to(config.device)
    print(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.T_max, 
                                                    eta_min=5e-8)
    model, history = run_training(model, optimizer, scheduler,device=config.device,num_epochs=config.max_epoch,train_loader=train_loader,valid_loader=valid_loader)

# %% [markdown]
# ## tasks

# %% [markdown]
# soybean task

# %%
soybean_task = {
    # ********************vist*****************
    "ViST":{
        "crop_name":"soybean",
        "model_name":"ViST",
        "wandb_name":"vist|大豆|290图像加传感器",
        "sensor_only":False,
        
    },
    "imageViST":{
        "crop_name":"soybean",
        "model_name":"imageViST",
        "wandb_name":"vist|大豆|290仅图片",
        "sensor_only":True,
    },
    "sensorViST":{
        "crop_name":"soybean",
        "model_name":"sensorViST",
        "wandb_name":"vist|大豆|290仅传感器",
        "sensor_only":True,
    },
    # ********************vist2***************** self attn和cross attn混用
    "ViST2":{
        "crop_name":"soybean",
        "model_name":"ViST2",
        "wandb_name":"vist|大豆|290图像加传感器",
        "sensor_only":False,
        
    },

    # ********************vilt*****************
    "sensorViLOnlyTransformerSS":{
        "crop_name":"soybean",
        "model_name":"sensorViLOnlyTransformerSS",
        "wandb_name":"vilt|大豆|290仅图片",
        "sensor_only":True,
    },
    "sensorOnlyViLTransformerSS":{
        "crop_name":"soybean",
        "model_name":"sensorOnlyViLTransformerSS",
        "wandb_name":"vilt|大豆|290仅传感器",
        "sensor_only":True,
    },
    "sensorViLTransformerSS":{
        "crop_name":"soybean",
        "model_name":"sensorViLTransformerSS",
        "wandb_name":"vilt|大豆|290图像加传感器",
        "sensor_only":False,
    },

    # ********************DNNF1*****************
    "DNNF1":{
        "crop_name":"soybean",
        "model_name":"DNNF1",
        "wandb_name":"DNNF1|大豆|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF1PictureOnly":{
        "crop_name":"soybean",
        "model_name":"DNNF1PictureOnly",
        "wandb_name":"DNNF1|大豆|290仅图像",
        "sensor_only":True,
    },
    "DNNF1SensorOnly":{
        "crop_name":"soybean",
        "model_name":"DNNF1SensorOnly",
        "wandb_name":"DNNF1|大豆|290仅传感器",
        "sensor_only":True,
    },
    # ********************DNNF2*****************
    "DNNF2":{
        "crop_name":"soybean",
        "model_name":"DNNF2",
        "wandb_name":"DNNF2|大豆|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF2PictureOnly":{
        "crop_name":"soybean",
        "model_name":"DNNF2PictureOnly",
        "wandb_name":"DNNF2|大豆|290仅图像",
        "sensor_only":True,
    },
    "DNNF2SensorOnly":{
        "crop_name":"soybean",
        "model_name":"DNNF2SensorOnly",
        "wandb_name":"DNNF2|大豆|290仅传感器",
        "sensor_only":True,
    },

    "RiceFusionMLP":{
        "crop_name":"soybean",
        "model_name":"RiceFusionMLP",
        "wandb_name":"RiceFusionMLP|大豆|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusionCNN":{
        "crop_name":"soybean",
        "model_name":"RiceFusionCNN",
        "wandb_name":"RiceFusionCNN|大豆|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusion":{
        "crop_name":"soybean",
        "model_name":"RiceFusion",
        "wandb_name":"RiceFusion|大豆|290图像加传感器",
        "sensor_only":False,
    },
    
    "RiceTransformer":{
        "crop_name":"soybean",
        "model_name":"RiceTransformer",
        "wandb_name":"RiceTransformer|大豆|290图像加传感器",
        "sensor_only":False,
    },
    "CNNTransformer":{
        "crop_name":"soybean",
        "model_name":"CNNTransformer",
        "wandb_name":"CNNTransformer|大豆|290图像加传感器",
        "sensor_only":False,
    },
    "SemanticEstimation":{
        "crop_name":"soybean",
        "model_name":"SemanticEstimation",
        "wandb_name":"SemanticEstimation|大豆|290图像加传感器",
        "sensor_only":False,
    },

    

}


# %% [markdown]
# corn task

# %%
corn_task = {
    # ********************vist*****************
    "ViST":{
        "crop_name":"corn",
        "model_name":"ViST",
        "wandb_name":"vist|玉米|290图像加传感器",
        "sensor_only":False,
        
    },
    "imageViST":{
        "crop_name":"corn",
        "model_name":"imageViST",
        "wandb_name":"vist|玉米|290仅图片",
        "sensor_only":True,
    },
    "sensorViST":{
        "crop_name":"corn",
        "model_name":"sensorViST",
        "wandb_name":"vist|玉米|290仅传感器",
        "sensor_only":True,
    },
    # ********************vist2***************** self attn和cross attn混用
    "ViST2":{
        "crop_name":"corn",
        "model_name":"ViST2",
        "wandb_name":"vist|玉米|290图像加传感器",
        "sensor_only":False,
        
    },

    # ********************vilt*****************
    "sensorViLOnlyTransformerSS":{
        "crop_name":"corn",
        "model_name":"sensorViLOnlyTransformerSS",
        "wandb_name":"vilt|玉米|290仅图片",
        "sensor_only":True,
    },
    "sensorOnlyViLTransformerSS":{
        "crop_name":"corn",
        "model_name":"sensorOnlyViLTransformerSS",
        "wandb_name":"vilt|玉米|290仅传感器",
        "sensor_only":True,
    },
    "sensorViLTransformerSS":{
        "crop_name":"corn",
        "model_name":"sensorViLTransformerSS",
        "wandb_name":"vilt|玉米|290图像加传感器",
        "sensor_only":False,
    },

    # ********************DNNF1*****************
    "DNNF1":{
        "crop_name":"corn",
        "model_name":"DNNF1",
        "wandb_name":"DNNF1|玉米|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF1PictureOnly":{
        "crop_name":"corn",
        "model_name":"DNNF1PictureOnly",
        "wandb_name":"DNNF1|玉米|290仅图像",
        "sensor_only":True,
    },
    "DNNF1SensorOnly":{
        "crop_name":"corn",
        "model_name":"DNNF1SensorOnly",
        "wandb_name":"DNNF1|玉米|290仅传感器",
        "sensor_only":True,
    },
    # ********************DNNF2*****************
    "DNNF2":{
        "crop_name":"corn",
        "model_name":"DNNF2",
        "wandb_name":"DNNF2|玉米|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF2PictureOnly":{
        "crop_name":"corn",
        "model_name":"DNNF2PictureOnly",
        "wandb_name":"DNNF2|玉米|290仅图像",
        "sensor_only":True,
    },
    "DNNF2SensorOnly":{
        "crop_name":"corn",
        "model_name":"DNNF2SensorOnly",
        "wandb_name":"DNNF2|玉米|290仅传感器",
        "sensor_only":True,
    },

    "RiceFusionMLP":{
        "crop_name":"corn",
        "model_name":"RiceFusionMLP",
        "wandb_name":"RiceFusionMLP|玉米|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusionCNN":{
        "crop_name":"corn",
        "model_name":"RiceFusionCNN",
        "wandb_name":"RiceFusionCNN|玉米|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusion":{
        "crop_name":"corn",
        "model_name":"RiceFusion",
        "wandb_name":"RiceFusion|玉米|290图像加传感器",
        "sensor_only":False,
    },
    
    "RiceTransformer":{
        "crop_name":"corn",
        "model_name":"RiceTransformer",
        "wandb_name":"RiceTransformer|玉米|290图像加传感器",
        "sensor_only":False,
    },
    "CNNTransformer":{
        "crop_name":"corn",
        "model_name":"CNNTransformer",
        "wandb_name":"CNNTransformer|玉米|290图像加传感器",
        "sensor_only":False,
    },
    "SemanticEstimation":{
        "crop_name":"corn",
        "model_name":"SemanticEstimation",
        "wandb_name":"SemanticEstimation|玉米|290图像加传感器",
        "sensor_only":False,
    },

    

}


# %% [markdown]
# rice

# %%
rice_task = {
    # ********************vist*****************
    "ViST":{
        "crop_name":"rice",
        "model_name":"ViST",
        "wandb_name":"vist|水稻|290图像加传感器",
        "sensor_only":False,
        
    },
    "imageViST":{
        "crop_name":"rice",
        "model_name":"imageViST",
        "wandb_name":"vist|水稻|290仅图片",
        "sensor_only":True,
    },
    "sensorViST":{
        "crop_name":"rice",
        "model_name":"sensorViST",
        "wandb_name":"vist|水稻|290仅传感器",
        "sensor_only":True,
    },
    # ********************vist2***************** self attn和cross attn混用
    "ViST2":{
        "crop_name":"rice",
        "model_name":"ViST2",
        "wandb_name":"vist|水稻|290图像加传感器",
        "sensor_only":False,
        
    },

    # ********************vilt*****************
    "sensorViLOnlyTransformerSS":{
        "crop_name":"rice",
        "model_name":"sensorViLOnlyTransformerSS",
        "wandb_name":"vilt|水稻|290仅图片",
        "sensor_only":True,
    },
    "sensorOnlyViLTransformerSS":{
        "crop_name":"rice",
        "model_name":"sensorOnlyViLTransformerSS",
        "wandb_name":"vilt|水稻|290仅传感器",
        "sensor_only":True,
    },
    "sensorViLTransformerSS":{
        "crop_name":"rice",
        "model_name":"sensorViLTransformerSS",
        "wandb_name":"vilt|水稻|290图像加传感器",
        "sensor_only":False,
    },

    # ********************DNNF1*****************
    "DNNF1":{
        "crop_name":"rice",
        "model_name":"DNNF1",
        "wandb_name":"DNNF1|水稻|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF1PictureOnly":{
        "crop_name":"rice",
        "model_name":"DNNF1PictureOnly",
        "wandb_name":"DNNF1|水稻|290仅图像",
        "sensor_only":True,
    },
    "DNNF1SensorOnly":{
        "crop_name":"rice",
        "model_name":"DNNF1SensorOnly",
        "wandb_name":"DNNF1|水稻|290仅传感器",
        "sensor_only":True,
    },
    # ********************DNNF2*****************
    "DNNF2":{
        "crop_name":"rice",
        "model_name":"DNNF2",
        "wandb_name":"DNNF2|水稻|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF2PictureOnly":{
        "crop_name":"rice",
        "model_name":"DNNF2PictureOnly",
        "wandb_name":"DNNF2|水稻|290仅图像",
        "sensor_only":True,
    },
    "DNNF2SensorOnly":{
        "crop_name":"rice",
        "model_name":"DNNF2SensorOnly",
        "wandb_name":"DNNF2|水稻|290仅传感器",
        "sensor_only":True,
    },

    "RiceFusionMLP":{
        "crop_name":"rice",
        "model_name":"RiceFusionMLP",
        "wandb_name":"RiceFusionMLP|水稻|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusionCNN":{
        "crop_name":"rice",
        "model_name":"RiceFusionCNN",
        "wandb_name":"RiceFusionCNN|水稻|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusion":{
        "crop_name":"rice",
        "model_name":"RiceFusion",
        "wandb_name":"RiceFusion|水稻|290图像加传感器",
        "sensor_only":False,
    },
    
    "RiceTransformer":{
        "crop_name":"rice",
        "model_name":"RiceTransformer",
        "wandb_name":"RiceTransformer|水稻|290图像加传感器",
        "sensor_only":False,
    },
    "CNNTransformer":{
        "crop_name":"rice",
        "model_name":"CNNTransformer",
        "wandb_name":"CNNTransformer|水稻|290图像加传感器",
        "sensor_only":False,
    },
    "SemanticEstimation":{
        "crop_name":"rice",
        "model_name":"SemanticEstimation",
        "wandb_name":"SemanticEstimation|水稻|290图像加传感器",
        "sensor_only":False,
    },

    

}


# %% [markdown]
# total task

# %%
total_task = {
    # ********************vist*****************
    "ViST":{
        "crop_name":"total",
        "model_name":"ViST",
        "wandb_name":"vist|普适|290图像加传感器",
        "sensor_only":False,
        
    },
    "imageViST":{
        "crop_name":"total",
        "model_name":"imageViST",
        "wandb_name":"vist|普适|290仅图片",
        "sensor_only":True,
    },
    "sensorViST":{
        "crop_name":"total",
        "model_name":"sensorViST",
        "wandb_name":"vist|普适|290仅传感器",
        "sensor_only":True,
    },
    # ********************vist2***************** self attn和cross attn混用
    "ViST2":{
        "crop_name":"total",
        "model_name":"ViST2",
        "wandb_name":"vist|普适|290图像加传感器",
        "sensor_only":False,
        
    },

    # ********************vilt*****************
    "sensorViLOnlyTransformerSS":{
        "crop_name":"total",
        "model_name":"sensorViLOnlyTransformerSS",
        "wandb_name":"vilt|普适|290仅图片",
        "sensor_only":True,
    },
    "sensorOnlyViLTransformerSS":{
        "crop_name":"total",
        "model_name":"sensorOnlyViLTransformerSS",
        "wandb_name":"vilt|普适|290仅传感器",
        "sensor_only":True,
    },
    "sensorViLTransformerSS":{
        "crop_name":"total",
        "model_name":"sensorViLTransformerSS",
        "wandb_name":"vilt|普适|290图像加传感器",
        "sensor_only":False,
    },

    # ********************DNNF1*****************
    "DNNF1":{
        "crop_name":"total",
        "model_name":"DNNF1",
        "wandb_name":"DNNF1|普适|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF1PictureOnly":{
        "crop_name":"total",
        "model_name":"DNNF1PictureOnly",
        "wandb_name":"DNNF1|普适|290仅图像",
        "sensor_only":True,
    },
    "DNNF1SensorOnly":{
        "crop_name":"total",
        "model_name":"DNNF1SensorOnly",
        "wandb_name":"DNNF1|普适|290仅传感器",
        "sensor_only":True,
    },
    # ********************DNNF2*****************
    "DNNF2":{
        "crop_name":"total",
        "model_name":"DNNF2",
        "wandb_name":"DNNF2|普适|290图像加传感器",
        "sensor_only":False,
    },
    "DNNF2PictureOnly":{
        "crop_name":"total",
        "model_name":"DNNF2PictureOnly",
        "wandb_name":"DNNF2|普适|290仅图像",
        "sensor_only":True,
    },
    "DNNF2SensorOnly":{
        "crop_name":"total",
        "model_name":"DNNF2SensorOnly",
        "wandb_name":"DNNF2|普适|290仅传感器",
        "sensor_only":True,
    },

    "RiceFusionMLP":{
        "crop_name":"total",
        "model_name":"RiceFusionMLP",
        "wandb_name":"RiceFusionMLP|普适|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusionCNN":{
        "crop_name":"total",
        "model_name":"RiceFusionCNN",
        "wandb_name":"RiceFusionCNN|普适|290图像加传感器",
        "sensor_only":False,
    },
    "RiceFusion":{
        "crop_name":"total",
        "model_name":"RiceFusion",
        "wandb_name":"RiceFusion|普适|290图像加传感器",
        "sensor_only":False,
    },
    
    "RiceTransformer":{
        "crop_name":"total",
        "model_name":"RiceTransformer",
        "wandb_name":"RiceTransformer|普适|290图像加传感器",
        "sensor_only":False,
    },
    "CNNTransformer":{
        "crop_name":"total",
        "model_name":"CNNTransformer",
        "wandb_name":"CNNTransformer|普适|290图像加传感器",
        "sensor_only":False,
    },
    "BilinearPooling":{
        "crop_name":"total",
        "model_name":"BilinearPooling",
        "wandb_name":"BilinearPooling|普适|290图像加传感器",
        "sensor_only":False,
    },
    "CompactBilinearPoolingRMP":{
        "crop_name":"total",
        "model_name":"CompactBilinearPoolingRMP",
        "wandb_name":"CompactBilinearPoolingRMP|普适|290图像加传感器",
        "sensor_only":False,
    },
    "CompactBilinearPoolingTSP":{
        "crop_name":"total",
        "model_name":"CompactBilinearPoolingTSP",
        "wandb_name":"CompactBilinearPoolingTSP|普适|290图像加传感器",
        "sensor_only":False,
    },
    "SemanticEstimation":{
        "crop_name":"total",
        "model_name":"SemanticEstimation",
        "wandb_name":"SemanticEstimation|普适|290图像加传感器",
        "sensor_only":False,
    },

    

}


# %% [markdown]
# run task
# 

# %%
import torch

# 假设有两个相同的模型 model1 和 model2
model1 = torch.load('/home/junsheng/ViLT/wandb/offline-run-20240316_152633-13gjsygd/files/epoch-1-13gjsygd.bin')
model2 =  torch.load('/home/junsheng/ViLT/wandb/offline-run-20240316_152633-13gjsygd/files/epoch-2-13gjsygd.bin')

# 加载模型参数

# model1.load_state_dict()
# model2.load_state_dict()

# 遍历模型中的每一层
for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    # 检查两个模型对应层的参数是否相等
    if torch.equal(param1.data, param2.data):
        print("相同的层名称:", name1)

# %% [markdown]
# #  test
# 

# %%
def test():
    model = build_model("ViST",True)
    state_dict = torch.load('/home/junsheng/ViLT/wandb/run-20230111_141431-tb52bngc/files/epoch-best.bin')
    model.load_state_dict(state_dict)
    model.to(config.device)
    test_df = pd.read_csv("/home/junsheng/ViLT/data/ubiquitous_soybean.csv")
    tianhang_sensor = ['co2', 'stemp', 'stemp2', 'stemp3', 'stemp5', 'shumi', 'shumi2', 'shumi3', 'shumi5', 'humi', 'pm10', 'pm25', 'press', 'solar', 'temp', 'wind_d', 'wind_sp']
    test_df['sensor'] = test_df[tianhang_sensor].values.tolist()
    test_data = BuildDataset(df=test_df,label=True)
    test_loader = DataLoader(test_data, batch_size=config.valid_batch_size,shuffle=False)
    val_loss,val_loss_mae,val_loss_smape,val_loss_mape = valid_one_epoch(model,test_loader,device=config.device) # # epoch_loss,epoch_loss_mae,epoch_loss_smape,epoch_loss_mape


