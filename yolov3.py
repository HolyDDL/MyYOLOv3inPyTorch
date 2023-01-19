from torch import nn as nn
import numpy as np
import torch
from torchvision.ops import nms
class DarkNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        # layers 2
        self.resblock1 = nn.Sequential(
            nn.Conv2d(64,32,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        # layers 6
        self.resblock2 = nn.Sequential(
            nn.Conv2d(128,64,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        # layers 13
        self.resblock3 = nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        # layers 38
        self.resblock4 = nn.Sequential(
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        # layers 63
        self.resblock5 = nn.Sequential(
            nn.Conv2d(1024,512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        
    def forward(self,input):
        # layer1->5
        x = self.conv1(input)
        y = self.resblock1(x)
        res = x + y 
        x = self.conv2(res)
        # layer5->12
        for i in range(2):
            y = self.resblock2(x)
            x = x + y
            x = nn.functional.leaky_relu(x)
        x = self.conv3(x)
        # layer13->37
        for i in range(8):
            y = self.resblock3(x)
            x = x + y
            x = nn.functional.leaky_relu(x)
        out1 = x
        x = self.conv4(x)
        # layer38->62
        for i in range(8):
            y = self.resblock4(x)
            x = x + y
            x = nn.functional.leaky_relu(x)
        out2 = x
        x = self.conv5(x)
        # layer63->74
        for i in range(4):
            y = self.resblock5(x)
            x = x + y
            x = nn.functional.leaky_relu(x) 
        out3 = x
        return out1, out2, out3

class Yolov3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # scale 1
        self.fore_scale1 = nn.Sequential(
            nn.Conv2d(1024,512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.mid_conv1 = nn.Sequential(
            nn.Conv2d(1024,512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.rear_scale1 = nn.Sequential(
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        # scale 2
        self.upsamle2 = nn.Sequential(
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.mid_scale2 = nn.Sequential(
            nn.Conv2d(768,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.mid_conv2 = nn.Sequential(
            nn.Conv2d(512,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.rear_scale2 = nn.Sequential(
            nn.Conv2d(256,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        # scale 3
        self.upsamle3 = nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.mid_scale3 = nn.Sequential(
            nn.Conv2d(384,128,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.mid_scale3_2 = nn.Sequential(
            nn.Conv2d(256,128,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.rear_scale3 = nn.Sequential(
            nn.Conv2d(256,256,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
    def forward(self, input1, input2, input3):
        '''
            parameters:

                input3 = 13*13*1024
                input2 = 26*26*512
                input1 = 52*52*256
        '''
        # scale 1
        x = input3
        for i in range(2):
            x = self.fore_scale1(x)
        x = self.mid_conv1(x)
        # layer79 out
        scale1_1 = x
        x = self.rear_scale1(x)
        # layer82 out
        scale1_out = x

        # scale 2
        x = scale1_1
        x = self.upsamle2(x)
        x = torch.cat((x,input2),dim=1)
        x = self.mid_scale2(x)
        x = self.mid_conv2(x)
        # layer91 out
        scale2_1 = x
        x = self.rear_scale2(x)
        # layer94 out
        scale2_out = x

        # scale 3
        x = scale2_1
        x = self.upsamle3(x)
        x = torch.cat((x,input1),dim=1)
        x = self.mid_scale3(x)
        for i in range(2):
            x = self.mid_scale3_2(x)
        x = self.rear_scale3(x)
        # layer106 out
        scale3_out = x
        return scale1_out, scale2_out, scale3_out

class Attribute(nn.Module):
    # 输出特征层
    def __init__(self,num_classes,num_anchors) -> None:
        '''
            Parameters:
                num_classes: 类别数量
                num_anchors: 单个网格生成几个锚框
        '''
        super().__init__()
        self.attr = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            # coco数据集80个类, 3*(80+5)=255
            nn.Conv2d(256,num_anchors*(num_classes+5),1),
            nn.BatchNorm2d(num_anchors*(num_classes+5)),
            nn.LeakyReLU()
        )
    
    def forward(self, input):
        return self.attr(input)
       
# if __name__ == '__main__':
#     dark = DarkNet()
#     input = torch.rand(1,3,416,416)
#     d1, d2, d3 = dark(input)
#     yolo = Yolov3()
#     out = yolo(d1,d2,d3)
#     At = Attribute(80,3)
#     attr = At(out[0])
#     dbox = DecodeBox(80,(416,416))
#     decode_anchors = dbox.decode_box(attr)
#     dbox.non_max_supperssion(decode_anchors, (1024, 916),0.5, 0.4)