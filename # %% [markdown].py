# %% [markdown]
# # import

# %%
import pytorch_lightning as pl
import torch
import torch.nn as nn
from vilt.modules import heads, objectives, vilt_utils
import vilt.modules.vision_transformer as vit
from typing import OrderedDict
import os
import pandas as pd
import numpy as np
from vilt.transforms import pixelbert_transform
from PIL import Image
from torchvision import transforms, utils
import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

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
    max_epoch = 100
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
    num_workers = 8
    precision = 16

config = vars(config)
config = dict(config)
config

# %%


# %%
df = pd.DataFrame({"sensor":[np.random.randn(10)]*10,"image_path":"assets/vilt.png","label":np.random.randint(1,10+1)})
df

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
        if self.label:
            label = self.labels[index]

            return torch.tensor(img), torch.tensor(sensor),torch.tensor(label)
        else:
            return torch.tensor(img), torch.tensor(sensor)

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

class sensorViLTransformerSS(pl.LightningModule):

    def __init__(self, config,sensor_class_n,output_class_n):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config["hidden_size"]) 

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # if self.config["load_path"] == "":
        self.transformer = getattr(vit, self.config["vit"])(
                pretrained=False, config=self.config
            )
       

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)
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
 
        sensor_embeds = self.sensor_linear(batch['sensor']) # input[1,1,12]  output[1,1,768]
        

        if image_embeds is None and image_masks is None:
            img = batch["image"][0]
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
        sensor_masks = batch['sensor_masks'] # 序列数量

        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x) # torch.Size([1, 240, 768])
        sensor_feats, image_feats = ( # torch.Size([1, 23, 768]),torch.Size([1, 217, 768])
            x[:, : sensor_embeds.shape[1]], # 后面字数输出23维
            x[:, sensor_embeds.shape[1] :], # 前面图片输出217维
        )
        cls_feats = self.pooler(x) # torch.Size([1, 768])
        cls_output = nn.Softmax(self.classifier(cls_feats))
        
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

    def training_step(self, batch, batch_idx):
        # vilt_utils.set_task(self)
        output = self(batch)
        loss = self.loss(output['cls_output'], x)

        return loss

    def training_epoch_end(self, outs):
        # vilt_utils.epoch_wrapup(self)
        pass

    def validation_step(self, batch, batch_idx):
        # vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        # vilt_utils.epoch_wrapup(self)
        pass
    def test_step(self, batch, batch_idx):
        # vilt_utils.set_task(self)
        # output = self(batch)
        # ret = dict()
        # return ret
        pass

    def test_epoch_end(self, outs):
        model_name = self.config["load_path"].split("/")[-1][:-5]

        # if self.config["loss_names"]["vqa"] > 0:
        #     objectives.vqa_test_wrapup(outs, model_name)
        # vilt_utils.epoch_wrapup(self)


    def configure_optimizers(self):
        """定义优化器
        """
        # return vilt_utils.set_schedule(self)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %% [markdown]
# ## model build

# %%
model = sensorViLTransformerSS(config,sensor_class_n= 12,output_class_n = 9)

# %% [markdown]
# # train

# %%
trainer = pl.Trainer(
    gpus=config["num_gpus"],
    # num_nodes=config["num_nodes"], # number of GPU nodes for distributed training.
    # precision=config["precision"], #  Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.
    # accelerator="ddp",
    # benchmark=True, # If true enables cudnn.benchmark.
    # deterministic=True,
    max_epochs=config["max_epoch"],
    # max_steps=config["max_steps"],
    # callbacks=callbacks, # Add a list of callbacks.
    # logger=logger,
    # prepare_data_per_node=False,
    # replace_sampler_ddp=False,
    # accumulate_grad_batches=grad_steps,
    # log_every_n_steps=10,
    # flush_logs_every_n_steps=10,
    # resume_from_checkpoint=config["resume_from"],
    # weights_summary="top",
    # fast_dev_run=config["fast_dev_run"],
    # val_check_interval=config["val_check_interval"],
    
)

# %%
# if not _config["test_only"]:
trainer.fit(model,train_dataloader=train_loader,val_dataloaders=valid_loader)
# else:
    # trainer.test(model, datamodule=dm)

# %%


# %% [markdown]
# # infer

# %%
# torch.save(model.state_dict(), 'embedding_test_dict.pt')
print(model)
model.setup("test")
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

    batch = {"text": [""], "image": [None]}
    batch["image"][0] = img

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
            "a display of flowers growing out and over the [MASK] [MASK] in front of [MASK] on a [MASK] [MASK].",
            0,
        ],
        [
            "6212487_1cca7f3f_1024x1024.jpg",
            "a a a display of flowers growing out and over the retaining wall in front of cottages on a cloudy day",
            4,
        ],
    ],

n = 1
sensor = torch.randn(1,1,12)
out = infer(examples[0][n][0],sensor)
# print("out:",out,"000\n")
# print("out0.shape:",out[0].shape)
# cv2.imwrite('output.png',out[0])



# %%
out


