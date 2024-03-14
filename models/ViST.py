import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch

class ViST(nn.Module):
    """图形加传感器

    Args:
        nn (_type_): _description_
    """
    def __init__(self,sensor_class_n,output_class_n,config):
        """图形加传感器

        Args:
            sensor_class_n (int): 输入的传感器数量
            output_class_n (int): 输出类别，回归问题则输出为1
            config (class): 配置信息
        """
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 
        self.sensor_linear2 = nn.Linear(1,145)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        # self.transformer = getattr(vit, config.vit)(
        #         pretrained=False, config=vars(config)
        #     )
        self.transformer = vit.VisionTransformerForViST(img_size=config.img_size,patch_size=config.patch_size,embed_dim=config.hidden_size,depth=config.num_layers,num_heads=config.num_heads,mlp_ratio=config.mlp_ratio,qkv_bias=False,qk_scale=None)
        
       
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
        sensor_embeds = torch.transpose(sensor_embeds,1,2)
        sensor_embeds = self.sensor_linear2(sensor_embeds)
        sensor_embeds = torch.transpose(sensor_embeds,1,2)

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(self.config.device)
            (
                image_embeds, # torch.Size([1, 217, 768])
                image_masks, # torch.Size([1, 217])
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.config.max_image_len,
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
        
        x_image = image_embeds # torch.Size([2, 145, 768])
        x_sensor = sensor_embeds # torch.Size([2, 145, 768])
        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(self.config.device)
            x_image,x_sensor, _attn_image,_attn_sensor = blk(x_image, x_sensor) # co_masks = torch.Size([1, 211])

        x = torch.cat([x_image, x_sensor], dim=1)
        x = self.transformer.norm(x) # torch.Size([1, 240, 768])

        cls_feats = self.pooler(x) # torch.Size([1, 768])
        # cls_feats = self.dense(x)
        # cls_feats = self.activation(cls_feats)
        cls_output = self.classifier(cls_feats)
        # m = nn.Softmax(dim=1)
        
        m = nn.Sigmoid()
        cls_output = m(cls_output)
        
        
        ret = {
           "sensor_feats":x_image,
            "image_feats": x_sensor,
            "cls_feats": cls_feats, # class features
            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret
    
class ViST2(nn.Module):
    """图像加传感器 self attention + cross attention一起用

    Args:
        nn (_type_): _description_
    """
    def __init__(self,sensor_class_n,output_class_n,config):
        """图像加传感器 self attention + cross attention一起用

        Args:
            sensor_class_n (int): 输入的传感器数量
            output_class_n (int): 输出类别，回归问题则输出为1
            config (class): 配置信息
        """
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 
        self.sensor_linear2 = nn.Linear(1,145)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        # self.transformer = getattr(vit, config.vit)(
        #         pretrained=False, config=vars(config)
        #     )
        self.transformer = vit.VisionTransformerForViST(img_size=config.img_size,patch_size=config.patch_size,embed_dim=config.hidden_size,num_heads=config.num_heads,mlp_ratio=config.mlp_ratio,qkv_bias=False,qk_scale=None)
        self.transformer_self_attn = getattr(vit, config.vit)(
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
        sensor_embeds = torch.transpose(sensor_embeds,1,2)
        sensor_embeds = self.sensor_linear2(sensor_embeds)
        sensor_embeds = torch.transpose(sensor_embeds,1,2)

        # 传感器数据self attention
        sensor_embeds_self = sensor_embeds
        x = sensor_embeds_self.to(self.config.device) # torch.Size([1, 1, 768])
        for i, blk in enumerate(self.transformer_self_attn.blocks):
            blk = blk.to(self.config.device)
            x, _attn = blk(x, mask=None)
        output_sensor = self.transformer_self_attn.norm(x)
        

        


        if image_embeds is None and image_masks is None:
            img = batch["image"].to(self.config.device)
            (
                image_embeds, # torch.Size([1, 217, 768])
                image_masks, # torch.Size([1, 217])
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.config.max_image_len,
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
        
        # 图像 self-attention
        image_embeds_self = image_embeds
        x = image_embeds_self.to(self.config.device) # torch.Size([1, 1, 768])
        for i, blk in enumerate(self.transformer_self_attn.blocks):
            blk = blk.to(self.config.device)
            x, _attn = blk(x, mask=None)
        output_image = self.transformer_self_attn.norm(x)

        # cross-attention
        x_image = image_embeds # torch.Size([2, 145, 768])
        x_sensor = sensor_embeds # torch.Size([2, 145, 768])
        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(self.config.device)
            x_image,x_sensor, _attn_image,_attn_sensor = blk(x_image, x_sensor) # co_masks = torch.Size([1, 211])

        x = torch.cat([x_image, x_sensor,output_image,output_sensor], dim=1)
        x = self.transformer.norm(x) # torch.Size([1, 240, 768])

        cls_feats = self.pooler(x) # torch.Size([1, 768])
        # cls_feats = self.dense(x)
        # cls_feats = self.activation(cls_feats)
        cls_output = self.classifier(cls_feats)
        if self.output_class_n == 1:
            m = nn.Sigmoid()
        else:
            m = nn.Softmax(dim=1)
        cls_output = m(cls_output)
        
        
        ret = {
           "sensor_feats":x_image,
            "image_feats": x_sensor,
            "cls_feats": cls_feats, # class features
            "cls_output":cls_output,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret


class sensorViST(nn.Module):
    """仅传感器

    Args:
        nn (_type_): _description_
    """
    def __init__(self,sensor_class_n,output_class_n,config):
        """仅传感器

        Args:
            sensor_class_n (int): 输入的传感器数量
            output_class_n (int): 输出类别，回归问题则输出为1
            config (class): 配置信息
        """
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_class_n,config.hidden_size) 
        self.sensor_linear2 = nn.Linear(1,145)
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
        sensor = batch['sensor'].to(self.config.device)
        sensor_embeds = self.sensor_linear(sensor) # input[1,1,12]  output[1,1,768]
        sensor_embeds = torch.transpose(sensor_embeds,1,2)
        sensor_embeds = self.sensor_linear2(sensor_embeds)
        sensor_embeds = torch.transpose(sensor_embeds,1,2)

        

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
        sensor_masks = torch.ones(sensor_embeds.shape[1],1).to(self.config.device) # 序列数量
        # image_masks = image_masks.to(config.device)
        # co_embeds = torch.cat([sensor_embeds, image_embeds], dim=1) # torch.Size([1, 240, 768]) ->240=217+23
        # co_masks = torch.cat([sensor_masks, image_masks], dim=1) # torch.Size([1, 240])
        co_embeds = sensor_embeds
        co_masks = sensor_masks

        x = co_embeds.to(self.config.device) # torch.Size([1, 1, 768])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(self.config.device)
            x, _attn = blk(x, mask=None)

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
    

class imageViST(nn.Module):
    """仅图像

    Args:
        nn (_type_): _description_
    """
    def __init__(self, sensor_class_n, output_class_n,config):
        """仅图像

        Args:
            sensor_class_n (int): 输入的传感器数量
            output_class_n (int): 输出类别，回归问题则输出为1
            config (class): 配置信息
        """
        super().__init__()
        self.config = config
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
            img = batch["image"].to(self.config.device)

            (
                image_embeds,  # torch.Size([1, 217, 768])
                image_masks,  # torch.Size([1, 217])
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.config.max_image_len,
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
        sensor_masks = torch.ones(batch_size, 1).to(self.config.device)  # 序列数量
        image_masks = image_masks.to(self.config.device)
        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds.to(self.config.device)  # torch.Size([1, 145, 768])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(self.config.device)
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