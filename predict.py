import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from net import Net_v1,Net_v2,LinearNet,MLP


#Net_v1这个网络实测太差，不知道哪的问题
# class Predictor_v1:
#     def __init__(self, weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.model = Net_v1().to(self.device)
#
#         # 加载训练好的模型权重
#         if os.path.exists(weight_path):
#             self.model.load_state_dict(torch.load(weight_path))
#             self.model.eval()  # 设置模型为评估模式
#         else:
#             raise FileNotFoundError(f"未找到模型权重文件: {weight_path}")
#
#     def predict(self, image_path):
#         # 加载和预处理图像
#         transform = transforms.Compose([
#             transforms.Grayscale(),  # 确保图像为灰度图
#             transforms.Resize((28, 28)),  # 调整大小以匹配模型输入
#             transforms.ToTensor(),  # 转换为张量
#             transforms.Normalize((0.5,), (0.5,))  # 归一化
#         ])
#
#         # 打开图像
#         img = Image.open(image_path)
#         img = transform(img).reshape(-1, 1 * 28 * 28).to(self.device)  # 添加批次维度并移动到设备
#
#         # 进行预测
#         with torch.no_grad():  # 禁用梯度计算
#             output = self.model(img)
#             predicted_label = torch.argmax(output, dim=1).item()  # 获取预测的类别标签
#
#         return predicted_label

class Predictor_v2:
    def __init__(self, weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = Net_v2().to(self.device)

        # 加载训练好的模型权重
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path))
            self.model.eval()  # 设置模型为评估模式
        else:
            raise FileNotFoundError(f"未找到模型权重文件: {weight_path}")

    def predict(self, image_path):
        # 加载和预处理图像
        transform = transforms.Compose([
            transforms.Grayscale(),  # 确保图像为灰度图
            transforms.Resize((28, 28)),  # 调整大小以匹配模型输入
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])

        # 打开图像
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0).to(self.device)  # 添加批次维度并移动到设备

        # 进行预测
        with torch.no_grad():  # 禁用梯度计算
            output = self.model(img)
            predicted_label = torch.argmax(output, dim=1).item()  # 获取预测的类别标签

        return predicted_label


class Predict_liner():
    def __init__(self, weight_path=""):
        self.weight_path = weight_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = LinearNet().to(self.device)
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))
            self.model.eval()  # 设置为评估模式
        else:
            raise FileNotFoundError(f"模型权重文件 {self.weight_path} 不存在")

        # 定义数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 调整为28x28像素
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    def predict(self, image_path):
        # 加载并预处理图片
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        image = self.transform(image).unsqueeze(0)  # 添加一个维度，变为 (1, 1, 28, 28)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, dim=1)

        return predicted.item()  # 返回预测的类别

class Predict_MLP():
    def __init__(self, weight_path=""):
        self.weight_path = weight_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = MLP().to(self.device)
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path))
            self.model.eval()  # 设置为评估模式
        else:
            raise FileNotFoundError(f"模型权重文件 {self.weight_path} 不存在")

        # 定义数据预处理
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),  # 调整为28x28像素
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        # 加载并预处理图片
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        image = self.transform(image).unsqueeze(0)  # 添加一个维度，变为 (1, 1, 28, 28)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            predicted = torch.argmax(outputs, dim=1)

        return predicted.item()  # 返回预测的类别
# 示例用法
if __name__ == "__main__":
    weight_path = './weights/mnist_15.pth'  # 指定训练好的模型路径
    #weight_path = './weights2/mnist_15.pth'  # 指定训练好的模型路径
    image_path = './datasets/5/000001.png'  # 指定要预测的图像路径

    predictor = Predictor_v1(weight_path)
    #predictor = Predictor_v2(weight_path)
    #predictor = Predict_liner(weight_path)
    predicted_label = predictor.predict(image_path)
    print(f'预测标签: {predicted_label}')