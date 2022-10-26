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


# %% [markdown]
# # config

# %%


class config:
    debug = False
    exp_name = "vilt"
    seed = 101
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    train_batch_size = 32
    valid_batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_fold = 5

    # wandb 
    wandb_name = "vilt|大豆|290图片加传感器"
    

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
    learning_rate = 1e-3
    weight_decay = 1e-4 # 0.01 ->1e-4
    decay_power = 1
    max_epoch = 50
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    # T_max = 8000/train_batch_size*max_epoch 
    T_max = 1000/train_batch_size*max_epoch 

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

# config = vars(config)
# config = dict(config)
config

if config.debug:
    config.max_epoch = 5

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
os.environ["WANDB_MODE"] = 'dryrun' # 离线模式
try:
    # wandb.log(key="*******") # if debug
    wandb.login() # storage in ~/.netrc file
    anonymous = None
except:
    anonymous = "must"
    print('\nGet your W&B access token from here: https://wandb.ai/authorize\n')


# %% [markdown]
# # 数据

# %%
df_tianhang = pd.read_csv("/home/junsheng/ViLT/data/290-tianhang-soybean.csv")
df_tianhang['image_path'] = df_tianhang['pic_key'].map(lambda x:os.path.join('/home/junsheng/data/tianhang_soybean',x.split('/')[-1]))
df_tianhang['label'] = df_tianhang['LAI']
df_tianhang = df_tianhang.dropna()
df_tianhang = df_tianhang.reset_index()
print(df_tianhang.shape)
df_tianhang.to_csv("test.csv",index=False)
df_tianhang.head()

# %% [markdown]
# 数据检查

# %%
# 检查图片下载的全不全
pic = df_tianhang.image_path.map(lambda x:x.split('/')[-1]).unique()
print(len(pic))
file_ls = os.listdir("/home/junsheng/data/tianhang_soybean")
print(len(file_ls))
ret = list(set(pic) ^ set(file_ls))
print(len(ret)) #差集
# assert len(pic)==len(file_ls),"请检查下载的图片，缺了{}个".format(len(pic)-len(file_ls))


# %% [markdown]
# 归一化非object列

# %%
list(df_tianhang)

# %%
number_title = []
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
        df_tianhang[title] = df_tianhang[title].map(lambda x:(x-x_min)/(x_max - x_min))
print(number_title)
print(recorder)

# %%
df_tianhang['stemp4'].dtype

# %%
# xiangguan_sensor = ['temperature', 'humidity', 'illuminance', 'soil_temperature', 'soil_humidity', 'pressure', 'wind_speed', 'photosynthetic', 'sun_exposure_time', 'COz', 'soil_ph']
tianhang_sensor = ['co2', 'stemp', 'stemp2', 'stemp3', 'stemp4', 'stemp5', 'shumi', 'shumi2', 'shumi3', 'shumi4', 'shumi5', 'humi', 'pm10', 'pm25', 'press', 'solar', 'temp', 'wind_d', 'wind_sp']
# tianhang_sensor = ['co2', 'stemp', 'stemp2', 'stemp3', 'stemp5', 'shumi', 'shumi2', 'shumi3', 'shumi5', 'humi', 'pm10', 'pm25', 'press', 'solar', 'temp', 'wind_d', 'wind_sp']

df_tianhang['sensor'] = df_tianhang[tianhang_sensor].values.tolist()
print("input dim:",len(tianhang_sensor))

# %%
df=df_tianhang
if config.debug:
    df = df[:100]
df.shape

# %%
df_tianhang.to_csv("test.csv",index=False)

# %% [markdown]
# create folds

# %%
skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)  
for fold, (train_idx, val_idx) in enumerate(skf.split(df,df.date)):
    df.loc[val_idx, 'fold'] = fold
df.groupby(['fold'])['label'].count()# ???

# %%
df.head()
df.to_csv("test_fold.csv",index=False)

# %% [markdown]
# # dataset
# 

# %%
myTransforms = transforms.Compose([
    transforms.Resize((config.img_size,config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.7136, 0.7118, 0.6788],
    std=[0.3338, 0.3453, 0.3020],
    
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
def fetch_dataloader(fold:int):
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


# %%
# train_dataset = BuildDataset(df=df)
# train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,shuffle=True)
# valid_loader = DataLoader(train_dataset, batch_size=config.valid_batch_size,shuffle=True)
train_loader,valid_loader = fetch_dataloader(fold=0)


# %%
img,sensor,label = next(iter(train_loader))
print(img.shape)
print(sensor.shape)
print(label.shape)

# %% [markdown]
# # model

# %% [markdown]
# sensorViLTransformerSS

# %%

class sensorViLTransformerSS(nn.Module):

    def __init__(self,sensor_class_n,output_class_n):
        super().__init__()
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        self.transformer = getattr(vit, config.vit)(
                pretrained=True, config=vars(config)
            )
       
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


        self.pooler = heads.Pooler(config.hidden_size)

        # self.pooler.apply(objectives.init_weights)
        self.classifier = nn.Linear(config.hidden_size,output_class_n)

        hs = config.hidden_size


    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        sensor = batch['sensor'].to(config.device)
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,12]  output[1,1,768]
        

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(config.device)
       
            (
                image_embeds, # torch.Size([1, 217, 768])
                image_masks, # torch.Size([1, 217])
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=config.max_image_len,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        # 用embedding对数据输入预处理，降低维度
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )
        # sensor_masks = batch['sensor_masks'] # 序列数量
        batch_size = img.shape[0]
        sensor_masks = torch.ones(batch_size,1).to(config.device) # 序列数量
        image_masks = image_masks.to(config.device)
        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])

        x = co_embeds.to(config.device) # torch.Size([1, 211, 768])

        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(config.device)
            x, _attn = blk(x, mask=co_masks) # co_masks = torch.Size([1, 211])

        x = self.transformer.norm(x) # torch.Size([1, 240, 768])
        sensor_feats, image_feats = ( # torch.Size([1, 23, 768]),torch.Size([1, 217, 768])
            x[:, : sensor_embeds.shape[1]], # 后面字数输出23维
            x[:, sensor_embeds.shape[1] :], # 前面图片输出217维
        )
        cls_feats = self.pooler(x) # torch.Size([1, 768])
        # cls_feats = self.dense(x)
        # cls_feats = self.activation(cls_feats)
        cls_output = self.classifier(cls_feats)
        # m = nn.Softmax(dim=1)
        
        m = nn.Sigmoid()
        cls_output = m(cls_output)
        
        ret = {
           "sensor_feats":sensor_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats, # class features
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
           
            "patch_index": patch_index,

            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret


# %% [markdown]
# sensorOnlyViLTransformerSS

# %%

class sensorOnlyViLTransformerSS(nn.Module):

    def __init__(self,sensor_class_n,output_class_n):
        super().__init__()
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        self.transformer = getattr(vit, config.vit)(
                pretrained=True, config=vars(config)
            )
       
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


        self.pooler = heads.Pooler(config.hidden_size)

        # self.pooler.apply(objectives.init_weights)
        self.classifier = nn.Linear(config.hidden_size,output_class_n)

        hs = config.hidden_size


    def infer(
        self,
        batch,
        # mask_image=False,
        # image_token_type_idx=1,
        # image_embeds=None,
        # image_masks=None,
    ):
        sensor = batch['sensor'].to(config.device)
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,12]  output[1,1,768]
        

        # if image_embeds is None and image_masks is None:
        #     img = batch["image"].to(config.device)
       
        #     (
        #         image_embeds, # torch.Size([1, 217, 768])
        #         image_masks, # torch.Size([1, 217])
        #         patch_index,
        #         image_labels,
        #     ) = self.transformer.visual_embed(
        #         img,
        #         max_image_len=config.max_image_len,
        #         mask_it=mask_image,
        #     )
        # else:
        #     patch_index, image_labels = (
        #         None,
        #         None,
        #     )
        # 用embedding对数据输入预处理，降低维度
        # image_embeds = image_embeds + self.token_type_embeddings(
        #         torch.full_like(image_masks, image_token_type_idx)
        #     )
        # sensor_masks = batch['sensor_masks'] # 序列数量
        # batch_size = img.shape[0]
        sensor_masks = torch.ones(sensor_embeds.shape[1],1).to(config.device) # 序列数量
        # image_masks = image_masks.to(config.device)
        # co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        # co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])
        co_embeds = sensor_embeds
        co_masks = sensor_masks

        x = co_embeds.to(config.device) # torch.Size([1, 1, 768])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(config.device)
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x) # torch.Size([1, 240, 768])
        # sensor_feats, image_feats = ( # torch.Size([1, 23, 768]),torch.Size([1, 217, 768])
        #     x[:, : sensor_embeds.shape[1]], # 后面字数输出23维
        #     x[:, sensor_embeds.shape[1] :], # 前面图片输出217维
        # )
        cls_feats = self.pooler(x) # torch.Size([1, 768])
        # cls_feats = self.dense(x)
        # cls_feats = self.activation(cls_feats)
        cls_output = self.classifier(cls_feats)
        # m = nn.Softmax(dim=1)
        
        m = nn.Sigmoid()
        cls_output = m(cls_output)
        
        ret = {
        #    "sensor_feats":sensor_feats,
            # "image_feats": image_feats,
            "cls_feats": cls_feats, # class features
            "raw_cls_feats": x[:, 0],
            # "image_labels": image_labels,
            # "image_masks": image_masks,
           
            # "patch_index": patch_index,

            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret


# %% [markdown]
# ## model build

# %%
# model = sensorOnlyViLTransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1)
model = sensorViLTransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1)
model.to(config.device)
print(config.device)
for i,m in enumerate(model.modules()):
    print(i,m)

# %% [markdown]
# test

# %%

# sensor = torch.rand(config.senser_input_num)
# # sensor = torch.ones(config.senser_input_num)
# print(sensor)
# sensor =  torch.tensor(sensor).unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 3])
# batch = {}
# batch['sensor'] = sensor
# batch['image'] = "/home/junsheng/data/xiangguan/pic/xiangguanD4-2021-05-24-10-00-25.jpeg"
# model(batch)

# %% [markdown]
# # 损失函数

# %%
criterion = F.mse_loss #均方误差损失函数
# criterion = F.mae_loss

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
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (img, sensor,label) in pbar:               
        
        
        batch_size = img.size(0)
        batch = {"image":img,"sensor":sensor}

        y_pred  = model(batch)
        label = label.to(config.device).unsqueeze(1)

        loss = criterion(y_pred['cls_output'], label)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss

# %% [markdown]
# # train

# %%

def run_training(model, optimizer, scheduler, device, num_epochs):
     # init wandb
    run = wandb.init(project="vilt",
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
        val_loss = valid_one_epoch(model,valid_loader,device=device,optimizer=optimizer)
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)

        wandb.log({"Train Loss": train_loss,
                    "Valid Loss": val_loss,
                "lr": scheduler.get_last_lr()[0]
                })
        if best_valid_loss > val_loss:
            best_valid_loss = val_loss
            # model_file_path = os.path.join(wandb.run.dir,"epoch-{}-{}.bin".format(epoch,wandb.run.id))
            model_file_path = os.path.join(wandb.run.dir,"epoch-best.bin")
            run.summary["Best Epoch"] = epoch
            torch.save(model.state_dict(), model_file_path)
            print("model save to", model_file_path)
            
    os.system("cp /home/junsheng/ViLT/my_vilt_tianhang_soybean.ipynb {}".format(wandb.run.dir))
    run.finish()
    return model, history

# %% [markdown]
# optimizer

# %%
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.T_max, 
                                                   eta_min=1e-5)


# %% [markdown]
# run train

# %%

model, history = run_training(model, optimizer, scheduler,device=config.device,num_epochs=config.max_epoch)



# %% [markdown]
# # infer

# %%
for (img,sensor,label) in valid_loader:
    print(img.shape,sensor.shape,label)
    break

# %%
# torch.save(model.state_dict(), 'embedding_test_dict.pt')
# print(model)

# model.load_state_dict(torch.load("/home/junsheng/ViLT/wandb/offline-run-20220811_120519-nzfb1xoz/files/epoch-best.bin"))
model.eval()
device = config.device
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

    batch = dict()
    batch["image"] = img

    batch['sensor_masks'] = torch.ones(1,1).to(device)
    with torch.no_grad():
        batch['sensor'] = sensor.to(device)       
        infer = model(batch)

        print(infer)
        sensor_emb, img_emb = infer["sensor_feats"], infer["image_feats"]# torch.Size([1, 23, 768]) torch.Size([1, 217, 768])
        cls_output = infer['cls_output']
        

    return [cls_output]


# %% [markdown]
# random test

# %%

examples=[
            "/home/junsheng/data/xiangguan/pic/xiangguanD4-2021-05-24-10-00-25.jpeg", #0
            
            "/home/junsheng/data/xiangguan/pic/xiangguanD4-2021-07-18-04-22-30-preset-18.jpeg", # 3
    ]



n = 1
sensor = torch.rand(config.senser_input_num)
# sensor = torch.ones(config.senser_input_num)
print(sensor)
sensor =  torch.tensor(sensor).unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 3])
out = infer(examples[0],sensor)
# print("out:",out,"000\n")
# print("out0.shape:",out[0].shape)
# cv2.imwrite('output.png',out[0])



# %%
out

# %%
print(out[0].cpu().numpy()[0][0])
#0.00031266143

# %% [markdown]
# test by valid

# %% [markdown]
# 选择三组生长期不同的数据去验证训练的结果

# %%
df_test = df.query("fold==0").reset_index(drop=True)
df_test.to_csv("test_by_valid.csv",index=False)
sensor_test_list = df_test.sensor.tolist()
image_test_list = df_test.image_path.tolist()

# %%
idx = 64
sensor =  torch.tensor(sensor_test_list[idx]).unsqueeze(0).unsqueeze(0)
out = infer(image_test_list[idx],sensor)

# %%
idx = 876
sensor =  torch.tensor(sensor_test_list[idx]).unsqueeze(0).unsqueeze(0)
out = infer(image_test_list[idx],sensor)

# %%
idx = 1817
sensor =  torch.tensor(sensor_test_list[idx]).unsqueeze(0).unsqueeze(0)
out = infer(image_test_list[idx],sensor)


