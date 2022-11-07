from collections import namedtuple
import torch
import os
from PIL import Image
from vilt.transforms import pixelbert_transform
import torch
import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit


import os

from vilt.transforms import pixelbert_transform
from PIL import Image



class config:
    debug = False
    exp_name = "vilt"
    seed = 101
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    train_batch_size = 32
    valid_batch_size = 4
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # root_path = r'E:\\Download\\xiangguan' # 存放数据的根目录
    root_path = r'/home/junsheng/data/xiangguan' # 存放数据的根目录
    n_fold = 5

    # wandb 
    wandb_name = "vilt|290 sensor only"
    

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
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,19]  output[1,1,768]
        

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(config.device)
       
            (
                image_embeds, # torch.Size([1, 210, 768])
                image_masks, # torch.Size([1, 210])
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
        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 211, 768]) ->211=210+1
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 211])

        x = co_embeds.to(config.device) # torch.Size([1, 211, 768])

        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(config.device)
            x, _attn = blk(x, mask=co_masks) # co_masks = torch.Size([1, 211])

        x = self.transformer.norm(x) # torch.Size([1, 211, 768])
        sensor_feats, image_feats = ( # torch.Size([1, 1, 768]),torch.Size([1, 210, 768])
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
        # ret = namedtuple("cls_output",cls_output.item())
        # ret = {
        # #    "sensor_feats":sensor_feats,
        # #     "image_feats": image_feats,
        # #     "cls_feats": cls_feats, # class features
        # #     "raw_cls_feats": x[:, 0],
        # #     "image_labels": image_labels,
        # #     "image_masks": image_masks,
           
        # #     "patch_index": patch_index,

        #     "cls_output":cls_output,
        # }

        return cls_output

    def forward(self, batch):
        return self.infer(batch)

if config.debug:
    config.max_epoch = 5
model = sensorViLTransformerSS(sensor_class_n= config.senser_input_num,output_class_n = 1)
def create_example():
    device = config.device
    sensor = torch.rand(config.senser_input_num)
    sensor =  torch.tensor(sensor).unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 3])

    img_path = os.path.join('pictures',"/home/junsheng/data/xiangguan/pic/xiangguanD4-2021-05-24-10-00-25.jpeg")
    image = Image.open(img_path).convert("RGB")
    img = pixelbert_transform(size=384)(image) # 将图像数据归一化torch.Size([3, 384, 576])
    img = torch.tensor(img)
    img = torch.unsqueeze(img, 0) # torch.Size([1, 3, 384, 576])
    img = img.to(device)
    batch = {}
    batch["image"] = img
    batch['sensor'] = sensor.to(device) 

    batch['sensor_masks'] = torch.ones(1,1).to(device)
    return batch


model = torch.jit.trace(model,create_example(),check_trace =False)
model.save("test_rice_model.torchscript")
if 1==0:
    # model = torch.load("test_rice_model.pth")
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
            cls_output = infer['cls_output']
            

        return [cls_output]



    examples=[
                "/home/junsheng/data/xiangguan/pic/xiangguanD4-2021-05-24-10-00-25.jpeg", #0
                
                "/home/junsheng/data/xiangguan/pic/xiangguanD4-2021-07-18-04-22-30-preset-18.jpeg", # 3
        ]



    n = 1
    sensor = torch.rand(config.senser_input_num)
    sensor =  torch.tensor(sensor).unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 3])
    out = infer(examples[0],sensor)
    print("output:",out)

