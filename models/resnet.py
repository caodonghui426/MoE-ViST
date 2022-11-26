import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch
import pretrainedmodels

class sensorResnet50TransformerSS(nn.Module):

    def __init__(self,sensor_class_n,output_class_n,config):
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 
        # resnet model
        resnet_model = pretrainedmodels.__dict__["resnet50"](
    num_classes=1000, pretrained='imagenet')
        features = list([resnet_model.conv1, resnet_model.bn1, resnet_model.relu, resnet_model.maxpool, resnet_model.layer1, resnet_model.layer2, resnet_model.layer3,resnet_model.layer4])
        conv = nn.Conv2d(2048, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        bn = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        relu = nn.ReLU(inplace=True)


        self.resnet_features = nn.Sequential(*features,conv,bn,relu)

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
        sensor = batch['sensor'].to(self.config.device)
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,12]  output[1,1,768]
        img = batch["image"].to(self.config.device)
        image_embeds = self.resnet_features(img) 
        image_embeds = image_embeds.flatten(2).transpose(1, 2)
        image_masks = torch.ones(image_embeds.shape[0],image_embeds.shape[1],dtype=torch.int64).to(self.config.device)

        # 用embedding对数据输入预处理，降低维度
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )
        # sensor_masks = batch['sensor_masks'] # 序列数量
        batch_size = img.shape[0]
        sensor_masks = torch.ones(batch_size,1).to(self.config.device) # 序列数量
        image_masks = image_masks.to(self.config.device)
        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])

        x = co_embeds.to(self.config.device) # torch.Size([1, 211, 768])

        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(self.config.device)
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
            "image_masks": image_masks,
           

            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret



class sensorResnet101TransformerSS(nn.Module):

    def __init__(self,sensor_class_n,output_class_n,config):
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 
        # resnet model
        resnet_model = pretrainedmodels.__dict__["resnet101"](
    num_classes=1000, pretrained='imagenet')
        features = list([resnet_model.conv1, resnet_model.bn1, resnet_model.relu, resnet_model.maxpool, resnet_model.layer1, resnet_model.layer2, resnet_model.layer3,resnet_model.layer4])
        conv = nn.Conv2d(2048, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
        bn = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        relu = nn.ReLU(inplace=True)


        self.resnet_features = nn.Sequential(*features,conv,bn,relu)

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
        sensor = batch['sensor'].to(self.config.device)
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,12]  output[1,1,768]
        img = batch["image"].to(self.config.device)
        image_embeds = self.resnet_features(img) 
        image_embeds = image_embeds.flatten(2).transpose(1, 2)
        image_masks = torch.ones(image_embeds.shape[0],image_embeds.shape[1],dtype=torch.int64).to(self.config.device)

        # 用embedding对数据输入预处理，降低维度
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )
        # sensor_masks = batch['sensor_masks'] # 序列数量
        batch_size = img.shape[0]
        sensor_masks = torch.ones(batch_size,1).to(self.config.device) # 序列数量
        image_masks = image_masks.to(self.config.device)
        co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])

        x = co_embeds.to(self.config.device) # torch.Size([1, 211, 768])

        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(self.config.device)
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
            "image_masks": image_masks,
           

            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret
