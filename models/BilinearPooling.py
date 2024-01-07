import torch
import torch.nn as nn
import torch.nn.functional as F
from vilt.modules import heads, objectives

import vilt.modules.vision_transformer as vit
class BilinearPooling(nn.Module):
    """双线性池化baseline

    Args:
        nn (_type_): _description_
    """
    def __init__(self,sensor_nums,config):
        super(BilinearPooling, self).__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_nums,config.hidden_size) 
        self.sensor_linear2 = nn.Linear(1,145)

        self.output_linear = nn.Linear(145,1)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        self.transformer = vit.VisionTransformerForViST(img_size=config.img_size,patch_size=config.patch_size,embed_dim=config.hidden_size,depth=config.num_layers,num_heads=config.num_heads,mlp_ratio=config.mlp_ratio,qkv_bias=False,qk_scale=None)


    def forward(self, batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,):
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
        x1 = x_image
        x2 = x_sensor

        # Reshape input tensors to have shape (batch_size, num_channels, -1)
        x1 = x1.view(x1.size(0), x1.size(1), -1)
        x2 = x2.view(x2.size(0), x2.size(1), -1)

        # Compute outer product
        outer_product = torch.einsum('bik,bjk->bij', x1, x2)

        # Sum along dimension 1 (element-wise addition)
        bilinear_pool = torch.sum(outer_product, dim=1)

        # Reshape to vector
        bilinear_pool = bilinear_pool.view(bilinear_pool.size(0), -1)

        # Sign square root and L2 normalization
        bilinear_pool = torch.sign(bilinear_pool) * torch.sqrt(torch.abs(bilinear_pool) + 1e-5)
        bilinear_pool = F.normalize(bilinear_pool, p=2, dim=1)

        x = self.output_linear(bilinear_pool)

        return {"cls_output":x}
