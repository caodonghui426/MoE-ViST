import torch.nn as nn
from models.RiceFusionMLP import RiceFusionMLP
from models.RiceFusionCNN import RiceFusionCNN
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch

class RiceFusion(torch.nn.Module):
    """对比模型，来自文章：Rice-fusion: A multimodality data fusion framework for rice disease diagnosis

    Args:
        torch (_type_): _description_
    """
 
    def __init__(self,sensor_nums,config):
        super(RiceFusion,self).__init__()
        self.config = config
        mlp = RiceFusionMLP(sensor_nums=sensor_nums,config=config)
        cnn = RiceFusionCNN(config=config)
        """        
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        """
        self.mlp_model = nn.Sequential(
            mlp.linear1,
            mlp.relu1,
            mlp.linear2,
            mlp.relu2
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
            cnn.linear2
        )

        self.linear1 = torch.nn.Linear(512+7, out_features= 20)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, out_features= 10)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(10, out_features= 1)

        





    def forward(self,batch):
        sensor_input = batch['sensor'].to(self.config.device)
        sensor_input = sensor_input.squeeze(dim=1)
        sensor_feature = self.mlp_model(sensor_input) # input:torch.Size([32, 19]),output:torch.Size([32, 7])

        img_input = batch["image"].to(self.config.device)

        img_feature = self.cnn_model(img_input)  # input:torch.Size([32, 3, 384, 384]) output:torch.Size([32, 512])

        co_feature = torch.cat([sensor_feature, img_feature], dim=1) #output:torch.Size([32, 519])
        x = self.linear1(co_feature)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return {"cls_output":x}
