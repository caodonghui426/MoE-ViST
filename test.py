# %% [markdown]
# # import

# %%
import torch
import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch.nn.functional as F
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

# %%
np.random.randn(10,10)

# %% [markdown]
# # config

# %%


class config:
    exp_name = "vilt"
    seed = 101
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    train_batch_size = 2
    valid_batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    img_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

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
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 3
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

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

config = vars(config)
config = dict(config)
config

# %%
def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        #os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(seed=666)

# %%
test_df = pd.DataFrame({
    "a":np.random.randn(500),
    "b":np.random.randn(500),
    "c":np.random.randn(500),
    "d":np.random.randn(500),
    "image_path":"assets/vilt.png",
    
})

# %%
test_df['label'] = test_df.a + 2*test_df.b + 3*test_df.c + 4*test_df.d
test_df['sensor'] = test_df[['a','b','c','d','a','b','c','d','a','b']].values.tolist()

# %%
test_df.head()
test_df.to_csv("test_df.csv")

# %%
# df = pd.DataFrame({"sensor":[np.random.randn(10)]*10,"image_path":"assets/vilt.png","label":np.random.randint(1,10+1)})
df=test_df

# %% [markdown]
# # dataset
# 

# %%
myTransforms = transforms.Compose([
    transforms.Resize((config["img_size"],config["img_size"])),
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
train_dataset = BuildDataset(df=df)
train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'],shuffle=True)
valid_loader = DataLoader(train_dataset, batch_size=config['valid_batch_size'],shuffle=True)


# %%
img,sensor,label = next(iter(train_loader))
print(img.shape)
print(sensor.shape)
print(label.shape)

# %% [markdown]
# # model

# %%

class sensorViLTransformerSS(nn.Module):

    def __init__(self, config,sensor_class_n,output_class_n):
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config["hidden_size"]) 

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # if self.config["load_path"] == "":
        self.transformer = getattr(vit, self.config["vit"])(
                pretrained=False, config=self.config
            )
       
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()


        self.pooler = heads.Pooler(config["hidden_size"])

        # self.pooler.apply(objectives.init_weights)
        self.classifier = nn.Linear(config["hidden_size"],output_class_n)
        # ===================== Downstream ===================== #
        # if (
        #     self.config["load_path"] != ""
        #     and not self.config["test_only"]
        # ):
        #     ckpt = torch.load(self.config["load_path"], map_location="cpu")
        #     if isinstance(ckpt,OrderedDict):

        #         state_dict = ckpt
        #     else:
        #         state_dict = ckpt["state_dict"]
        #     self.load_state_dict(state_dict, strict=False)

        hs = self.config["hidden_size"]

        # vilt_utils.set_metrics(self) # 设定模型评价

        # ===================== load downstream (test_only) ======================

        if self.config["load_path"] != "" and self.config["test_only"]:
            ckpt = torch.load(self.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        sensor = batch['sensor'].to(config['device'])
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,12]  output[1,1,768]
        

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(config['device'])
       
            (
                image_embeds, # torch.Size([1, 217, 768])
                image_masks, # torch.Size([1, 217])
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.config["max_image_len"],
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
        sensor_masks = torch.ones(batch_size,1).to(config['device']) # 序列数量
        image_masks = image_masks.to(config['device'])
        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])

        x = co_embeds.to(config['device'])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(config['device'])
            x, _attn = blk(x, mask=co_masks)

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
        # cls_output = m(cls_output)

        
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
# ## model build

# %%
model = sensorViLTransformerSS(config,sensor_class_n= 10,output_class_n = 1)
model.to(config['device'])
print(config['device'])

# %% [markdown]
# # 损失函数

# %%
criterion = F.mse_loss #均方误差损失函数

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
        label = label.to(config['device']).unsqueeze(1)
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
# # train

# %%
import gc
def run_training(model, optimizer, scheduler, device, num_epochs):
    history = defaultdict(list)
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)
        
        # val_loss, val_scores = valid_one_epoch(model, valid_loader, 
                                                #  device=CFG.device, 
                                                #  epoch=epoch)
        # val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        # history['Valid Loss'].append(val_loss)
        

        
        # deep copy the model
        # if val_dice >= best_dice:
            # print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            # best_dice    = val_dice
            # best_jaccard = val_jaccard
            # best_epoch   = epoch
            # run.summary["Best Dice"]    = best_dice
            # run.summary["Best Jaccard"] = best_jaccard
            # run.summary["Best Epoch"]   = best_epoch
            # best_model_wts = copy.deepcopy(model.state_dict())
            # PATH = os.path.join(CFG.model_output_path, f"best_epoch-{fold:02d}.bin")
            # torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            # wandb.save(PATH)
            # print(f"Model Saved{sr_} to path:",PATH)
            
        # last_model_wts = copy.deepcopy(model.state_dict())
        # PATH = os.path.join(CFG.model_output_path,f"last_epoch-{fold:02d}.bin")
        # torch.save(model.state_dict(), PATH)


    
    # load best model weights
    # model.load_state_dict(best_model_wts)
    
    return model, history

# %% [markdown]
# optimizer

# %%
optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0001)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=1000, 
                                                   eta_min=0.0001)


# %% [markdown]
# run train

# %%
model, history = run_training(model, optimizer, scheduler,
                                device=config['device'],
                                num_epochs=config['max_epoch'])

# %% [markdown]
# # infer

# %%
# torch.save(model.state_dict(), 'embedding_test_dict.pt')
print(model)
model.eval()
device = config["device"]
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

examples=[
        [
            "6212487_1cca7f3f_1024x1024.jpg",
        ],
        [
            "6212487_1cca7f3f_1024x1024.jpg",
        ],
        [
            "6212487_1cca7f3f_1024x1024.jpg",
        ],
    ],

n = 1
sensor = torch.randn(1,1,10)
out = infer(examples[0][n][0],sensor)
# print("out:",out,"000\n")
# print("out0.shape:",out[0].shape)
# cv2.imwrite('output.png',out[0])



# %%
out

# %%
out[0].cpu().numpy()[0][0]


