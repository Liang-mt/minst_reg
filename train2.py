# Copyright (c) 2023 ChenJun

import onnx
import os
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from net import MLP
Epoch_max = 100


def save_model(model):
    dummy_input = torch.randn(1, 28, 28, 1)
    torch.onnx.export(model, dummy_input, "mlp.onnx")
    onnx_model = onnx.load("mlp.onnx")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

class Train_v2:
    def __init__(self, train_data_path, test_data_path, weight_path="", epoch_num=50):
        self.weight_path = weight_path
        self.epoch_num = epoch_num
        self.summaryWriter = SummaryWriter("logs")
        self.net = MLP().to(device)

        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

        # 从指定路径加载数据集
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=train_data_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.RandomAffine(degrees=(-5, 5), translate=(0.08, 0.08), scale=(0.9, 1.1)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomErasing(scale=(0.02, 0.02))
            ])
        )
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=test_data_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor()
            ])
        )

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=100, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=100, shuffle=False)

    def __call__(self):
        index1, index2 = 0, 0
        for epoch in range(self.epoch_num):
            # 训练阶段
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(device)
                label = label.to(device)

                # 前向传播
                train_y = self.net(img)
                train_loss = self.loss_fn(train_y, label)

                # 反向传播
                self.opt.zero_grad()
                train_loss.backward()
                self.opt.step()

                if i % 100 == 0:
                    print(f'Epoch: {epoch + 1}, Batch: {i}, Train Loss: {train_loss.item()}')
                    self.summaryWriter.add_scalar("train_loss", train_loss.item(), index1)
                    index1 += 1

            # 测试阶段
            with torch.no_grad():
                for i, (img, label) in enumerate(self.test_loader):
                    img = img.to(device)
                    label = label.to(device)

                    test_y = self.net(img)
                    test_loss = self.loss_fn(test_y, label)

                    test_y = torch.argmax(test_y, dim=1)
                    acc = (test_y == label).float().mean()

                    if i % 10 == 0:
                        print(f'Epoch: {epoch + 1}, Batch: {i}, Test Loss: {test_loss.item()}, Accuracy: {acc.item()}')
                        self.summaryWriter.add_scalar("test_loss", test_loss.item(), index2)
                        index2 += 1

            # 每5个epoch保存一次模型
            if (epoch + 1) % 5 == 0:
                weight = f'./weights3/mlp_{epoch + 1}.pth'
                torch.save(self.net.state_dict(), weight)
                print("模型已保存。")

            # 每个epoch后保存ONNX模型
            save_model(self.net)
            print("\n")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用指定的数据集路径初始化训练
train = Train_v2(train_data_path='./data/data_train01', test_data_path='./data/data_test01', epoch_num=Epoch_max)
train()