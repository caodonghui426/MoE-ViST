import torch.nn as nn
from vilt.modules import heads, objectives
import vilt.modules.vision_transformer as vit
import torch


class RiceFusionCNN(torch.nn.Module):
    """对比模型，来自文章：Rice-fusion: A multimodality data fusion framework for rice disease diagnosis


    Args:
        torch (_type_): _description_
    """

    def __init__(self, config):
        super(RiceFusionCNN, self).__init__()
        self.config = config
        self.img_size = config.img_size

        self.conv1_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32,
            kernel_size=3, padding=1, stride=1
        )
        self.conv1_2 = torch.nn.Conv2d(
            in_channels=32, out_channels=32,
            kernel_size=3, padding=1, stride=1
        )

        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = torch.nn.Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=3, padding=1, stride=1
        )
        self.conv2_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64,
            kernel_size=3, padding=1, stride=1
        )
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = torch.nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, padding=1, stride=1
        )
        self.conv3_2 = torch.nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=3, padding=1, stride=1
        )
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        self.linear1 = torch.nn.Linear(in_features=int((config.img_size/8)**2 * 128), out_features= 1024)
        self.relu4 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(config.drop_rate)
        self.linear2 = torch.nn.Linear(in_features=1024, out_features= 512)
        self.relu5 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(config.drop_rate)
        self.linear3 = torch.nn.Linear(in_features=512, out_features= 1)








    def forward(self, batch):
        """_summary_

        Args:
            batch (dict): batch['image']-> torch.Size([32, 3, 384, 384])

        Returns:
            tensor: x.shape->torch.Size([32, 1])
        """

        img = batch['image'].to(self.config.device)

        x = self.conv1_1(img) # input:torch.Size([32, 3, 384, 384]),output:torch.Size([32, 32, 384, 384])
        x = self.conv1_2(x) # output:torch.Size([32, 32, 384, 384])
        x = self.relu1(x) # output:torch.Size([32, 32, 384, 384])
        x = self.pool1(x) # output:torch.Size([32, 32, 192, 192])

        x = self.conv2_1(x) # output:torch.Size([32, 64, 192, 192])
        x = self.conv2_2(x) # output:torch.Size([32, 64, 192, 192])
        x = self.relu2(x) # output:torch.Size([32, 64, 192, 192])
        x = self.pool2(x) # output:torch.Size([32, 64, 96, 96])

        x = self.conv3_1(x) # output:torch.Size([32, 128, 96, 96])
        x = self.conv3_2(x) # output:torch.Size([32, 128, 96, 96])
        x = self.relu3(x) # output:torch.Size([32, 128, 96, 96])
        x = self.pool3(x) # output:torch.Size([32, 128, 48, 48])

        x  = torch.nn.Flatten()(x)

        x = self.linear1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.linear3(x) # output: torch.Size([32, 1])


        return {"cls_output": x}
