import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch

class sensorViLOnlyTransformerSS(nn.Module):
    """sensorViLOnlyTransformerSS-仅vit

    Args:
        nn (_type_): _description_
    """
    def __init__(self, sensor_class_n, output_class_n,config):
        super().__init__()
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        self.transformer = getattr(vit, config.vit)(
            pretrained=True, config=vars(config)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pooler = heads.Pooler(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, output_class_n)

    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(config.device)

            (
                image_embeds,  # torch.Size([1, 217, 768])
                image_masks,  # torch.Size([1, 217])
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
        sensor_masks = torch.ones(batch_size, 1).to(config.device)  # 序列数量
        image_masks = image_masks.to(config.device)
        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds.to(config.device)  # torch.Size([1, 145, 768])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(config.device)
            x, _attn = blk(x, mask=co_masks)  # co_masks = torch.Size([1, 211])

        x = self.transformer.norm(x)  # torch.Size([1, 240, 768])
        image_feats = x
        cls_feats = self.pooler(x)  # torch.Size([1, 768])
        # cls_feats = self.dense(x)
        # cls_feats = self.activation(cls_feats)
        cls_output = self.classifier(cls_feats)
        # m = nn.Softmax(dim=1)

        m = nn.Sigmoid()
        cls_output = m(cls_output)

        ret = {
            "image_feats": image_feats,
            "cls_feats": cls_feats,  # class features
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,

            "patch_index": patch_index,

            "cls_output": cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()

        ret.update(self.infer(batch))
        return ret


class sensorViLTransformerSS(nn.Module):

    def __init__(self,sensor_class_n,output_class_n,config):
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


class sensorOnlyViLTransformerSS(nn.Module):

    def __init__(self,sensor_class_n,output_class_n,config):
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
