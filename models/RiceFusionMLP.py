import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch

class RiceFusionMLP(torch.nn.Module):
    """对比模型，来自文章：Rice-fusion: A multimodality data fusion framework for rice disease diagnosis

    Args:
        torch (_type_): _description_
    """
 
    def __init__(self,sensor_nums:int,config):
        """_summary_

        Args:
            sensor_nums (int): 传感器数量
            config (class): 配置信息类
        """
        super(RiceFusionMLP,self).__init__()
        self.config = config
        self.linear1=torch.nn.Linear(sensor_nums,4)
        self.relu1=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(4,7) 
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(7,4)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(4,1)
  
    def forward(self, batch):
        """_summary_

        Args:
            batch (dict): 主要取batch['sensor']，输入shape为[32, 1, 19]

        Returns:
            tensor: 输出shape为[32,1]
        """
        sensor_input = batch['sensor'].to(self.config.device) # test input shape:torch.Size([32, 1, 19])
        x = sensor_input.squeeze(dim=1)
        
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return {"cls_output":x}



