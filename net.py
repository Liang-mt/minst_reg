import torch
from torch import nn
import torch.nn.functional as F

#这个网络和屎一样
class Net_v1(nn.Module):  # 修正为 nn.Module
    # 初始化模型
    def __init__(self):
        super(Net_v1, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(1 * 28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )
    #前向计算
    def forward(self, x):
        return self.fc_layer(x)


class Net_v2(nn.Module):  # 修正为 nn.Module
    # 初始化模型
    def __init__(self):
        super(Net_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,16,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 44, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(44, 64, (3, 3)),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(64*7*7,10),
            nn.Softmax(dim=1)
        )
    #前向计算
    def forward(self, x):
        out = self.layers(x).reshape(-1,64*7*7)
        out = self.out(out)
        return out

##Minist的图像为1X28X28的
class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        ##784是图像为 1X28X28，通道X宽度X高度，然后总的输入就是784
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        ##定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, x):
        ##将输入的图像矩阵改为N行784列
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        ##最后一层激活在损失函数中加入了，这里直接输出，不要加上rule了
        return self.l5(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 1, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.layers(x)

# 使用Sequential定义的全连接网络（结构与FcModelMan相同）
class FcModel(nn.Module):
    def __init__(self):
        super(FcModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 展平层（Bx1x28x28 -> Bx784）
            nn.Linear(28 * 28, 256), nn.SiLU(),  # 全连接层+激活
            nn.Linear(256, 512), nn.SiLU(),  # 维度变化：256->512
            nn.Linear(512, 256), nn.SiLU(),  # 维度变化：512->256
            nn.Linear(256, 10),  # 输出层（10分类）
        )

    def forward(self, x):
        return self.model(x)  # 直接返回Sequential处理结果

if __name__ == '__main__':
    x = torch.randn(1, 1 * 28 * 28)
    x2 = torch.randn(1, 1, 28, 28)
    net = LinearNet()
    print(net(x).shape)