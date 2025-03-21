from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn,optim
from torch.nn.functional import one_hot
from net import Net_v1,Net_v2,LinearNet
import os
import time

train_dataset = datasets.MNIST("C:/Users/28645/Desktop/minst_reg/",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST("C:/Users/28645/Desktop/minst_reg/",train=False,transform=transforms.ToTensor(),download=True)

train_dataloader = DataLoader(train_dataset,batch_size=100,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=100,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Train_v1():
    def __init__(self,weight_path="",epoch_num =15):
        self.weight_path = weight_path
        self.epoch_num = epoch_num

        self.summaryWriter=SummaryWriter("logs")
        self.net = Net_v1().to(device)

        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = optim.Adam(self.net.parameters())
        self.fc_loss = nn.CrossEntropyLoss()
        self.train = True
        self.test = True

    def __call__(self):
        index1,index2=0,0
        for epoch in range(self.epoch_num):
            if self.train:
                for i,(img,label) in enumerate(train_dataloader):
                    label = label.to(device)  # 直接使用原始标签
                    img = img.reshape(-1,1*28*28).to(device)

                    train_y = self.net(img)
                    train_loss = self.fc_loss(train_y,label)

                    #清空梯度
                    self.opt.zero_grad()
                    #梯度计算
                    train_loss.backward()
                    #梯度更新
                    self.opt.step()

                    if i % 100 == 0:
                        print(f'{epoch + 1}-{i}-train_loss===>>{train_loss.item()}')
                        self.summaryWriter.add_scalar("train_loss",train_loss,index1)
                        index1+=1

                if (epoch + 1) % 5 == 0:
                    weight = f'./weights/mnist_{epoch + 1}.pth'  # 每5轮保存一次模型
                    torch.save(self.net.state_dict(), weight)  # 保存模型
                    print("模型已保存")

            if self.test:
                for i, (img, label) in enumerate(test_dataloader):
                    label = one_hot(label, 10).float().to(device)
                    img = img.reshape(-1, 1 * 28 * 28).to(device)

                    test_y = self.net(img)
                    test_loss = self.fc_loss(test_y, label)

                    test_y = torch.argmax(test_y,dim=1)
                    label = torch.argmax(label,dim=1)

                    acc = torch.mean(torch.eq(test_y,label).float())

                    if i % 10 == 0:
                        print(f'{epoch + 1}-{i}-test_loss===>>{test_loss.item()}')
                        print(f'acc--{i}===>>{acc.item()}')
                        self.summaryWriter.add_scalar("test_loss", test_loss, index2)
                        index2 += 1


class Train_v2():
    def __init__(self,weight_path="",epoch_num =50):
        self.weight_path = weight_path
        self.epoch_num = epoch_num

        self.summaryWriter=SummaryWriter("logs")
        self.net = Net_v2().to(device)

        if os.path.exists(weight_path):
            self.net.load_state_dict(torch.load(weight_path))
        self.opt = optim.Adam(self.net.parameters())
        self.fc_loss=nn.MSELoss()
        self.train = True
        self.test = True

    def __call__(self):
        index1,index2=0,0
        for epoch in range(self.epoch_num):
            if self.train:
                for i,(img,label) in enumerate(train_dataloader):
                    label = one_hot(label,10).float().to(device)
                    img = img.to(device)

                    train_y = self.net(img)
                    train_loss = self.fc_loss(train_y,label)

                    #清空梯度
                    self.opt.zero_grad()
                    #梯度计算
                    train_loss.backward()
                    #梯度更新
                    self.opt.step()

                    if i % 100 == 0:
                        print(f'{epoch + 1}-{i}-train_loss===>>{train_loss.item()}')
                        self.summaryWriter.add_scalar("train_loss",train_loss,index1)
                        index1+=1

                if (epoch + 1) % 5 == 0:
                    weight = f'./weights2/mnist_{epoch + 1}.pth'  # 每5轮保存一次模型
                    torch.save(self.net.state_dict(), weight)  # 保存模型
                    print("模型已保存")

            if self.test:
                for i, (img, label) in enumerate(test_dataloader):
                    label = one_hot(label, 10).float().to(device)
                    img = img.to(device)

                    test_y = self.net(img)
                    test_loss = self.fc_loss(test_y, label)

                    test_y = torch.argmax(test_y,dim=1)
                    label = torch.argmax(label,dim=1)

                    acc = torch.mean(torch.eq(test_y,label).float())

                    if i % 10 == 0:
                        print(f'{epoch + 1}-{i}-test_loss===>>{test_loss.item()}')
                        print(f'acc--{i}===>>{acc.item()}')
                        self.summaryWriter.add_scalar("test_loss", test_loss, index2)
                        index2 += 1


class TrainLinearNet():
    def __init__(self, weight_path="", epoch_num=10, batch_size=64, learnrate=0.01, momentnum=0.5):
        self.weight_path = weight_path
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.learnrate = learnrate
        self.momentnum = momentnum

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.summaryWriter = SummaryWriter("logs")

        # 数据准备
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])


        self.train_dataloader = train_dataloader


        self.test_dataloader = test_dataloader

        # 初始化模型
        self.model = LinearNet().to(self.device)
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))

        # 优化器
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learnrate, momentum=self.momentnum)
        self.criterion = self.model.criterion
        self.toppredicted = 0.0

    def train(self, epoch):
        running_loss = 0.0
        for batch_idx, (inputs, target) in enumerate(self.train_dataloader):
            inputs, target = inputs.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if batch_idx % 300 == 299:
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 300:.3f}')
                self.summaryWriter.add_scalar("train_loss", running_loss / 300,
                                              epoch * len(self.train_dataloader) + batch_idx)
                running_loss = 0.0

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        currentpredicted = (100 * correct / total)
        if currentpredicted > self.toppredicted:
            self.toppredicted = currentpredicted
            torch.save(self.model.state_dict(), './weights/LinearNet.pt')
            print(f'模型已保存为 LinearNet.pt，当前准确率: {currentpredicted:.2f}%')

        print(f'测试集准确率: {currentpredicted:.2f}%')

    def __call__(self):
        for epoch in range(self.epoch_num):
            self.train(epoch)
            self.test()


if __name__ == '__main__':
    train = Train_v1(epoch_num=15)
    train()


