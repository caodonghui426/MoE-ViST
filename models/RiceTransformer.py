import torch.nn as nn

from models.RiceFusionMLP import RiceFusionMLP
from models.RiceFusionCNN import RiceFusionCNN
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch


class RiceTransformer(torch.nn.Module):
    """对比模型，来自文章：Rice transformer: A novel integrated management system for controlling rice diseases

    Args:
        torch (_type_): _description_
    """
 
    def __init__(self,sensor_nums,config):
        super(RiceTransformer,self).__init__()
        self.config = config
        cnn = RiceFusionCNN(config=config)
        self.sensor_self_attn = SelfAttention(dim=128,num_heads=4)
        self.img_self_attn = SelfAttention(dim=128,num_heads=4)
        
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)

        self.output_linear = nn.Linear(128,1)
        self.cross_attn = CrossAttention(dim=128,num_heads=4)
        self.mlp_model = nn.Sequential(
            nn.Linear(sensor_nums,8),
            nn.ReLU(),
            nn.Linear(8,46),
            nn.ReLU(),
            nn.Linear(46,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128)
        )

        self.cnn_model = nn.Sequential(
            cnn.conv1_1,
            cnn.conv1_2,
            cnn.relu1,
            cnn.pool1,

            cnn.conv2_1,
            cnn.conv2_2,
            cnn.relu2,
            cnn.pool2,

            cnn.conv3_1,
            cnn.conv3_2,
            cnn.relu3,
            cnn.pool3,
            torch.nn.Flatten(),
            cnn.linear1,
            cnn.relu4,
            cnn.dropout2,
            cnn.linear2,
            nn.Linear(512,128)
        )



        





    def forward(self,batch):
        sensor_input = batch['sensor'].to(self.config.device)
        sensor_input = sensor_input.squeeze(dim=1)
        sensor_feature = self.mlp_model(sensor_input) # input:torch.Size([32, 19]),output:torch.Size([32, 128])

        img_input = batch["image"].to(self.config.device) # torch.Size([32, 3, 384, 384])
        img_feature = self.cnn_model(img_input) # torch.Size([32, 128])

        img_feature,_ = self.img_self_attn(torch.unsqueeze(img_feature,1))
        sensor_feature,_ = self.sensor_self_attn(torch.unsqueeze(sensor_feature,1))

        img_feature, sensor_feature, attn_image, attn_sensor = self.cross_attn(img_feature,sensor_feature)


        img_feature = self.pool(img_feature) # output:torch.Size([32, 1, 64])
        sensor_feature = self.pool(sensor_feature) # output:torch.Size([32, 1, 64])
        output_feature = torch.cat([img_feature,sensor_feature],dim=2)

        x = self.output_linear(output_feature)
        x = torch.squeeze(x,dim=1)
        return {"cls_output":x}




class CrossAttention(nn.Module):

    def __init__(
        self,
        dim,  # embedding dimension ，mine is 768
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 768/12=32
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_image = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_sensor = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_image, x_sensor):
        B, N, C = x_image.shape  # torch.Size([2, 146, 768])
        qkv_image = (  # torch.Size([3, 2, 12, 146, 64]) # output is attn_image
            self.qkv_image(x_image)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv_sensor = (  # torch.Size([3, 2, 12, 146, 64])
            self.qkv_sensor(x_sensor)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q_sensor, k_image, v_image = (
            qkv_image[0],  # torch.Size([2, 12, 146, 64])
            qkv_image[1],  # torch.Size([2, 12, 146, 64])
            qkv_image[2],  # torch.Size([2, 12, 146, 64])
        )  # make torchscript happy (cannot use tensor as tuple)
        q_image, k_sensor, v_sensor = (
            qkv_sensor[0],  # torch.Size([2, 12, 146, 64])
            qkv_sensor[1],  # torch.Size([2, 12, 146, 64])
            qkv_sensor[2],  # torch.Size([2, 12, 146, 64])
        )  # make torchscript happy (cannot use tensor as tuple)

        attn_image = (q_sensor @ k_image.transpose(-2, -1)) * \
            self.scale  # torch.Size([2, 12, 146, 146])
        attn_sensor = (q_image @ k_sensor.transpose(-2, -1)) * \
            self.scale  # torch.Size([2, 12, 146, 146])

        # torch.Size([2, 12, 146, 146])
        attn_image = attn_image.softmax(dim=-1)
        # torch.Size([2, 12, 146, 146])
        attn_sensor = attn_sensor.softmax(dim=-1)
        attn_image = self.attn_drop(attn_image)
        attn_sensor = self.attn_drop(attn_sensor)

        x_image = (attn_image @ v_image).transpose(1, 2).reshape(B, N, C)
        x_sensor = (attn_sensor @ v_sensor).transpose(1, 2).reshape(B, N, C)
        #v:torch.Size([2, 12, 146, 64])
        #x:torch.Size([2, 146, 768])
        x_image = self.proj(x_image)
        x_sensor = self.proj(x_sensor)
        x_image = self.proj_drop(x_image)
        x_sensor = self.proj_drop(x_sensor)
        return x_image, x_sensor, attn_image, attn_sensor
    

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,  # embedding dimension ，mine is 768
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 768/12=32
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape  # torch.Size([32, 1, 128])
        qkv = (  # torch.Size([3, 32, 4, 1, 32])
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],  # torch.Size([2, 12, 146, 64])
            qkv[1],  # torch.Size([2, 12, 146, 64])
            qkv[2],  # torch.Size([2, 12, 146, 64])
        )  # make torchscript happy (cannot use tensor as tuple)

        # torch.Size([2, 12, 146, 146])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask = mask.bool() # device:cuda,
            mask = mask.type(torch.bool)  # device:cuda,
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)  # torch.Size([2, 12, 146, 146])
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #v:torch.Size([2, 12, 146, 64])
        #x:torch.Size([2, 146, 768])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
