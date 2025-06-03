# 实验程序：Made by DDXie@43，实验日期：2025年5月
# 实验目的：用于学习计算机视觉中的“看懂”图片中的人脸表情，并标记出表情标签。

# 导入所需的库,用于加载训练好的Keras模型
from keras.models import load_model

# 用于将图像转换为数组格式（方便输入到模型中）
from keras.preprocessing.image import img_to_array

# NumPy库，用于进行数值计算和数组操作
import numpy as np

# OpenCV库，用于图像读取、处理和显示
import cv2

# 加载预训练的表情识别模型（XCEPTION架构）
# 模型文件路径为 'face_classification-master/trained_models/fer2013_big_XCEPTION.54-0.66.hdf5'
# compile=False 表示不重新编译模型，直接使用保存时的配置
model = load_model(
    'fer2013_big_XCEPTION.54-0.66.hdf5',
    compile=False # 跳过编译阶段
)

# 定义表情类别标签（顺序必须与训练模型时一致）
# FER2013 数据集中的7种基本表情：
expression_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# 分别对应：愤怒、厌恶、恐惧、高兴、悲伤、惊讶、中性表情

# 读取原始图像文件（请确保路径正确）
image = cv2.imread('001image.jpg')

# 将图像转换为灰度图
# 因为模型是基于灰度图像训练的（FER2013数据集是灰度图像）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用OpenCV内置的Haar级联分类器检测人脸
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# detectMultiScale方法用于检测图像中的所有人脸
# 返回值是一个包含每个人脸位置信息的列表，每个元素形式为 (x, y, w, h)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # 图像缩放比例，用于补偿距离摄像机远近不同的人脸大小
    minNeighbors=5,      # 检测框保留阈值，越高越准确但可能漏检
    minSize=(30, 30)     # 最小人脸尺寸
)

# 遍历每一张检测到的人脸
for (x, y, w, h) in faces:
    
    # 提取当前人脸区域（ROI）作为感兴趣区域
    face = gray[y:y+h, x:x+w]
    
    # 将人脸图像调整为模型所需的输入尺寸（64x64像素）
    face = cv2.resize(face, (64, 64))
    
    # 将图像像素值归一化到 [0, 1] 区间（与训练时一致）
    face = face.astype("float") / 255.0
    
    # 将图像转换为Keras所需的数组格式，形状为 (64, 64, 1)
    # 其中最后一个维度表示通道数（灰度图只有一个通道）
    face = img_to_array(face)
    
    # 扩展维度以适配模型输入要求
    # 模型期望输入形状为 (batch_size, height, width, channels)，所以需要增加一个维度
    # 转换后形状为 (1, 64, 64, 1)
    face = np.expand_dims(face, axis=0)
  
    # 使用模型进行预测，得到7个类别的概率分布
    predictions = model.predict(face)[0]
    
    # 获取预测概率最高的类别索引
    max_index = np.argmax(predictions)
    
    # 根据索引获取对应的表情标签
    emotion = expression_labels[max_index]
  
    # 在原图上绘制绿色矩形框标记人脸区域
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 在人脸框上方添加红色文字标注预测出的表情
    cv2.putText(
        image,
        emotion,                      # 要显示的文本内容
        (x, y - 10),                  # 文字左下角坐标（在人脸框上方留点空间）
        cv2.FONT_HERSHEY_SIMPLEX,     # 字体类型
        0.9,                          # 字体大小
        (0, 0, 255),                  # 文字颜色（红色）
        2                             # 文字粗细
    )

# 将带有表情标注的图像保存为新文件
cv2.imwrite('emotion_result.jpg', image)

# 显示带有表情标注的结果图像窗口
cv2.imshow('Emotion Detection', image)

# 等待任意按键按下后关闭所有OpenCV窗口
cv2.waitKey(0)
cv2.destroyAllWindows()