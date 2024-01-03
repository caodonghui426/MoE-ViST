import torch
import torch.nn as nn
import torch.nn.functional as F
from vilt.modules import heads, objectives

import vilt.modules.vision_transformer as vit
class CompactBilinearPoolingRMP(nn.Module):
    """双线性池化CBP算法1，Random Maclaurin Projection
        文章：Multimodal compact bilinear pooling for visual question answering and visual grounding
    Args:
        nn (_type_): _description_
    """

    def __init__(self,sensor_nums,config):
        super(CompactBilinearPoolingRMP, self).__init__()
        # Define the input dimension c and the output dimension d
        self.c = config.hidden_size
        self.d = config.RMP_d
        # Generate random but fixed W1 and W2, where each entry is either +1 or -1 with equal probability[^1^][1]
        self.w1 = (torch.randint(0, 2, (self.d, self.c)) * 2 - 1).float().to(config.device)
        self.w2 = (torch.randint(0, 2, (self.d, self.c)) * 2 - 1).float().to(config.device)


        self.config = config
        self.sensor_linear = nn.Linear(sensor_nums,config.hidden_size) 
        self.sensor_linear2 = nn.Linear(1,145)

        self.output_linear = nn.Linear(145,1)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        self.transformer = vit.VisionTransformerForViST(img_size=config.img_size,patch_size=config.patch_size,embed_dim=config.hidden_size,depth=config.num_layers,num_heads=config.num_heads,mlp_ratio=config.mlp_ratio,qkv_bias=False,qk_scale=None)

    # Define the feature map function
    def phi_RM(self,x,device):
        batch_size = x.shape[0]
        # Assume x is a 3D tensor of size [2, 145, c]
        x = x.reshape(-1, self.c)  # Reshape x to a 2D tensor of size [(2*145), c]
        z = torch.sqrt(torch.tensor(1.0 / self.d)) * (self.w1 @ x.t()).t() * (self.w2 @ x.t()).t().to(device) # x.t()是矩阵转置
        z = z.reshape(batch_size,z.shape[0]//batch_size,self.d) # shape[2,145,10000]
        return z
    
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

        x = x_image
        y = x_sensor
        z_x = self.phi_RM(x,self.config.device)  # The output feature tensor
        z_y = self.phi_RM(y,self.config.device)  # The output feature tensor
        inner_product = torch.sum(z_x * z_y, dim=2)


        x = self.output_linear(inner_product)

        return {"cls_output":x}
    def test(self):
        # Example usage
        x = torch.randn(2, 145, self.c)  # A random input tensor
        y = torch.randn(2, 145, self.c)  # A random input tensor
        z_x = self.phi_RM(x)  # The output feature tensor
        z_y = self.phi_RM(y)  # The output feature tensor

        # Compute the inner product of x and y along the c dimension
        inner_product = torch.sum(z_x * z_y, dim=2)

        print(inner_product.shape)  # Should print: torch.Size([2, 145])

