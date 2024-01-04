import torch
import torch.nn as nn
import torch.nn.functional as F
from vilt.modules import heads, objectives

import vilt.modules.vision_transformer as vit
class CompactBilinearPoolingTSP(nn.Module):
    """双线性池化CBP算法2，Tensor Sketch Projection
        文章：Multimodal compact bilinear pooling for visual question answering and visual grounding
    Args:
        nn (_type_): _description_
    """

    def __init__(self,sensor_nums,config):
        super(CompactBilinearPoolingTSP, self).__init__()
        # 定义输入向量的维度和输出向量的维度
        self.c = config.hidden_size
        self.d = config.RMP_d

        # 生成随机但固定的哈希函数和符号函数
        torch.manual_seed(0) # 设置随机种子，保证结果可复现
        self.h1 = torch.randint(1, self.d + 1, (self.c,)).to(config.device) # 生成一个长度为c的整数向量，每个元素在[1, d]之间
        self.h2 = torch.randint(1, self.d + 1, (self.c,)).to(config.device)
        self.s1 = torch.randint(0, 2, (self.c,)).to(config.device) * 2 - 1 # 生成一个长度为c的整数向量，每个元素为+1或-1
        self.s2 = torch.randint(0, 2, (self.c,)).to(config.device) * 2 - 1

        self.config = config
        self.sensor_linear = nn.Linear(sensor_nums,config.hidden_size) 
        self.sensor_linear2 = nn.Linear(1,145)

        self.output_linear = nn.Linear(145,1)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(objectives.init_weights)
        self.transformer = vit.VisionTransformerForViST(img_size=config.img_size,patch_size=config.patch_size,embed_dim=config.hidden_size,depth=config.num_layers,num_heads=config.num_heads,mlp_ratio=config.mlp_ratio,qkv_bias=False,qk_scale=None)


    # 定义sketch函数
    def sketch(self,x, h, s):
        # x: 输入向量，维度为[batch_size, seq_len, c]
        # h: 哈希函数，维度为[c]
        # s: 符号函数，维度为[c]
        # 返回: 输出向量，维度为[batch_size, seq_len, d]
        batch_size, seq_len, _ = x.shape
        Q = torch.zeros((batch_size, seq_len, self.d)).to(self.config.device) # 初始化Q矩阵，维度为[batch_size, seq_len, d]
        Q.scatter_add_(2, h.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.c) - 1, x * s.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.c)) # 根据h和s更新Q矩阵，使用scatter_add_函数实现
        return Q
    
    # 定义Tensor Sketch Projection函数
    def tensor_sketch_projection(self,x):
        # x: 输入向量，维度为[batch_size, seq_len, c]
        # 返回: 输出向量，维度为[batch_size, seq_len, d]
        Q1 = self.sketch(x, self.h1, self.s1) # 使用第一组哈希函数和符号函数计算sketch
        Q2 = self.sketch(x, self.h2, self.s2) # 使用第二组哈希函数和符号函数计算sketch
        return torch.fft.ifft(torch.fft.fft(Q1) * torch.fft.fft(Q2)).real # 使用快速傅里叶变换和逆变换计算输出向量，取实部

    
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

        #使用TSP算法计算两个向量的内积，用来替代原来的外积
        z_x = self.tensor_sketch_projection(x)
        z_y = self.tensor_sketch_projection(y)
        inner_product = torch.sum(z_x * z_y, dim=2)

        # Reshape to vector
        bilinear_pool = inner_product.reshape(inner_product.size(0), -1)

        # Sign square root and L2 normalization
        bilinear_pool = torch.sign(bilinear_pool) * torch.sqrt(torch.abs(bilinear_pool) + 1e-5)
        bilinear_pool = F.normalize(bilinear_pool, p=2, dim=1)

        x = self.output_linear(bilinear_pool)

        return {"cls_output":x}
    
    def test(self):
        # 生成随机的输入向量，维度为[2, 145, 768]
        x = torch.randn(2, 145, 768)
        y = torch.randn(2, 145, 768)

        # 调用Tensor Sketch Projection函数，得到输出向量，维度为[2, 145, 10000]
        z_x = self.tensor_sketch_projection(x)
        z_y = self.tensor_sketch_projection(y)

        # 打印输出向量的形状
        print(x.shape)
        print(y.shape)
        inner_product = torch.sum(z_x * z_y, dim=2)
        print(inner_product.shape)
