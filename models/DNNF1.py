import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch



class DNNF1(torch.nn.Module):
    """dnnf1 图片加传感器

    Args:
        torch (_type_): _description_
    """
 
    def __init__(self,sensor_nums,config):
        super(DNNF1,self).__init__()
        self.sensor_linear = torch.nn.Linear(sensor_nums,768)
        
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        self.transformer = getattr(vit, config.vit)(
                pretrained=True, config=vars(config)
            )
       
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


        self.pooler = heads.Pooler(config.hidden_size)


        # DNNF1结构
        self.linear1=torch.nn.Linear(768+768,64)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(64,128)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(128,256)
        self.relu3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(256,512)
        self.relu4=torch.nn.ReLU()
        self.linear5=torch.nn.Linear(512,512)
        self.relu5=torch.nn.ReLU()
        self.linear6=torch.nn.Linear(512,1024)
        self.relu6=torch.nn.ReLU()
        self.linear7=torch.nn.Linear(1024,1)


    def forward(self,batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,):
        sensor_input = batch['sensor'].to(config.device)
        sensor_feats = self.sensor_linear(sensor_input)

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(config.device) # torch.Size([1, 3, 384, 384])

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
        image_masks = image_masks.to(config.device)
        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds.to(config.device)  # torch.Size([1, 145, 768])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(config.device)
            x, _attn = blk(x, mask=co_masks)  # co_masks = torch.Size([1, 211])

        x = self.transformer.norm(x)  # torch.Size([1, 240, 768])
        picture_feats = self.pooler(x)  # torch.Size([1, 768])#图像的特征数据
        sensor_feats = sensor_feats.squeeze(dim=1) #torch.Size([1, 1, 768])->[1,768]

        x = torch.cat([picture_feats, sensor_feats], dim=1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        return {"cls_output":x}


class DNNF1PictureOnly(torch.nn.Module):
    """DNNF1 picture only


    Args:
        torch (_type_): _description_
    """
 
    def __init__(self,sensor_nums,config):
        super(DNNF1PictureOnly,self).__init__()
        self.config = config
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)

        self.transformer = getattr(vit, config.vit)(
                pretrained=True, config=vars(config)
            )
       
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


        self.pooler = heads.Pooler(config.hidden_size)


        # DNNF1结构
        self.linear1=torch.nn.Linear(768,64)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(64,128)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(128,256)
        self.relu3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(256,512)
        self.relu4=torch.nn.ReLU()
        self.linear5=torch.nn.Linear(512,512)
        self.relu5=torch.nn.ReLU()
        self.linear6=torch.nn.Linear(512,1024)
        self.relu6=torch.nn.ReLU()
        self.linear7=torch.nn.Linear(1024,1)


    def forward(self,batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,):

        if image_embeds is None and image_masks is None:
            img = batch["image"].to(self.config.device) # torch.Size([1, 3, 384, 384])

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
        image_masks = image_masks.to(self.config.device)
        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds.to(self.config.device)  # torch.Size([1, 145, 768])

        for i, blk in enumerate(self.transformer.blocks):
            blk = blk.to(self.config.device)
            x, _attn = blk(x, mask=co_masks)  # co_masks = torch.Size([1, 211])

        x = self.transformer.norm(x)  # torch.Size([1, 240, 768])
        picture_feats = self.pooler(x)  # torch.Size([1, 768])#图像的特征数据

        x = picture_feats

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        return {"cls_output":x}
    

class DNNF1SensorOnly(torch.nn.Module):
    """DNNF1 sensor only

    Args:
        torch (_type_): _description_
    """
 
    def __init__(self,sensor_nums,config):
        super(DNNF1SensorOnly,self).__init__()
        self.sensor_linear = torch.nn.Linear(sensor_nums,768)
        self.config = config
        # DNNF1结构
        self.linear1=torch.nn.Linear(768,64)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(64,128)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(128,256)
        self.relu3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(256,512)
        self.relu4=torch.nn.ReLU()
        self.linear5=torch.nn.Linear(512,512)
        self.relu5=torch.nn.ReLU()
        self.linear6=torch.nn.Linear(512,1024)
        self.relu6=torch.nn.ReLU()
        self.linear7=torch.nn.Linear(1024,1)


    def forward(self,batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,):
        sensor_input = batch['sensor'].to(self.config.device)
        sensor_feats = self.sensor_linear(sensor_input)

        sensor_feats = sensor_feats.squeeze(dim=1) #torch.Size([1, 1, 768])->[1,768]

        x = sensor_feats

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        return {"cls_output":x}