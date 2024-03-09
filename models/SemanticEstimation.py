import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch
import torch.nn.functional as F
class SemanticEstimation(nn.Module):
    """基于语义估计的图像与传感器特征融合方法模型框架

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

        self.transformer = getattr(vit, config.vit)(
                pretrained=False, config=vars(config)
            )
        # self.transformer = vit.VisionTransformerForViST(img_size=config.img_size,patch_size=config.patch_size,embed_dim=config.hidden_size,depth=config.num_layers,num_heads=config.num_heads,mlp_ratio=config.mlp_ratio,qkv_bias=False,qk_scale=None)
        
        self.mlp1 = MLP(290*290,768,2)
        self.mlp2 = MLP(290*290,768,2)
       
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
        multimodal_feature = torch.cat([x_image, x_sensor], dim=1) # torch.Size([32, 290, 768])
        for i, blk in enumerate(self.transformer.blocks): 
            blk = blk.to(self.config.device)
            multimodal_feature,multimodal_feature_attn = blk(multimodal_feature,mask=None) #attn:torch.Size([32, 12, 290, 290]) multimodal_feature:torch.Size([32, 290, 768])

        multimodal_feature_X = multimodal_feature
        # attn特征图处理
        multimodal_feature_attn = multimodal_feature_attn.sum(dim=1)
        attn_shape = multimodal_feature_attn.shape  
        multimodal_feature_attn = multimodal_feature_attn.view(attn_shape[0], -1)  # 形状变为 [32, 290*290]
        multimodal_feature_attn = F.softmax(multimodal_feature_attn, dim=1)
        # 把softmax后的结果变回原始的形状 [32, 290, 290]
        multimodal_feature_attn = multimodal_feature_attn.view(attn_shape)


        relative_semantic = multimodal_feature_attn
        relative_semantic_T = torch.transpose(multimodal_feature_attn, 1, 2)
        multimodal_semantic_weight1 = self.mlp1(relative_semantic)
        multimodal_semantic_weight2 = self.mlp2(relative_semantic_T)
        multimodal_semantic_weight = (multimodal_semantic_weight1 +multimodal_semantic_weight2)/2
        multimodal_semantic_weight = torch.softmax(multimodal_semantic_weight,dim=1)

        # ******************语义加权计算*********************
        image_feature, sensor_feature = multimodal_feature.split(145, dim=1)

        # 使用multimodal_semantic_weight中存储的权重信息
        # 进行加权
        image_feature_clone = image_feature.clone()
        sensor_feature_clone = sensor_feature.clone()
        for i in range(multimodal_semantic_weight.shape[0]):
            image_feature_clone[i] = image_feature[i] * multimodal_semantic_weight[i][0].item()
            sensor_feature_clone[i] = sensor_feature[i] * multimodal_semantic_weight[i][1].item()
        image_feature = image_feature_clone
        sensor_feature = sensor_feature_clone

        # 拼接image_feature和sensor_feature
        multimodal_feature = torch.cat((image_feature, sensor_feature), dim=1) # torch.Size([32, 290, 768])

        # 执行批量矩阵乘法
        multimodal_feature = torch.bmm(multimodal_feature.transpose(1, 2) , multimodal_feature_attn)  # result shape 为 [32, 768, 290]
        multimodal_feature = multimodal_feature.transpose(1, 2)  # 最终形状变为 [32, 290, 768]
        multimodal_feature_WY = multimodal_feature
        multimodal_feature_Z = multimodal_feature_WY + multimodal_feature_X


        # *****************对比语义估计******************


        x = self.transformer.norm(multimodal_feature_Z) # torch.Size([1, 240, 768])

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
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the input tensor if needed
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
