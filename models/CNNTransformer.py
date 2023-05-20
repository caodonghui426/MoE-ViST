import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch

class CNNTransformer(nn.Module):

    def __init__(self,sensor_nums,config):
        super().__init__()
        self.config = config
        self.sensor_linear = nn.Linear(sensor_nums,config.img_size * config.img_size) 
        self.conv = nn.Conv2d(4,3,3,1,1)
        self.transformer = getattr(vit, config.vit)(
                pretrained=True, config=vars(config)
        )
        self.linear = nn.Linear(145*768,1)

       
    def infer(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        
        sensor = batch['sensor'].to(self.config.device)
        sensor_feature = self.sensor_linear(sensor)
        batch_size = sensor.shape[0]
        sensor_feature = sensor_feature.view(batch_size,1,384,384)

        img = batch["image"].to(self.config.device)
        img_feature = img
        x = torch.cat([img_feature,sensor_feature],dim=1)
        x = self.conv(x)
        (
            x, 
            image_masks, 
            patch_index,
            image_labels,
        ) = self.transformer.visual_embed(
            x,
            max_image_len=self.config.max_image_len,
            mask_it=mask_image,
        )
        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(self.config.device)
            x, _attn = blk(x)
        x = x.flatten(start_dim=1,end_dim=2)
        x = self.linear(x)
        ret = {
            "cls_output":x
        }

        return ret

    def forward(self, batch):
        ret = dict()
        
        ret.update(self.infer(batch))
        return ret