import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L').resize((28, 28))
    image_array = np.array(image).astype(np.float32) / 255.0
    # 调整为形状 (1, 28, 28, 1)
    return np.expand_dims(np.expand_dims(image_array, axis=-1), axis=0)

# 加载ONNX模型
onnx_model_path = "mlp_2.onnx"
onnx_model = onnx.load(onnx_model_path)
ort_session = ort.InferenceSession(onnx_model_path)

# 进行预测
def predict(image_path):
    # 预处理图像
    image = preprocess_image(image_path)

    # 进行推理
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)

    # 获取预测结果
    predicted_class = np.argmax(ort_outs[0], axis=1)
    return predicted_class[0]

# 示例：对新图像进行预测
image_path = './datasets/5/000000.png'  # 替换为你的图像路径
predicted_label = predict(image_path)
print(f'Predicted label: {predicted_label}')


