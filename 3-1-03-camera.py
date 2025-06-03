# 实验程序：Made by DDXie@43，实验日期：2025年5月
# 实验目的：用于学习计算机视觉中的“看懂”视频中的人脸表情，并标记出表情标签。

# 导入所需的库
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# 加载预训练的表情识别模型（XCEPTION架构）
model = load_model('fer2013_big_XCEPTION.54-0.66.hdf5', compile=False)

# 定义表情类别标签
expression_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 使用OpenCV内置的Haar级联分类器检测人脸
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头
camera = cv2.VideoCapture(0)

while True:
    # 读取一帧图像
    ret, frame = camera.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # 遍历每一张检测到的人脸
    for (x, y, w, h) in faces:
        # 提取当前人脸区域（ROI）作为感兴趣区域
        face = gray[y:y+h, x:x+w]
        
        # 调整尺寸和归一化等预处理步骤
        face = cv2.resize(face, (64, 64))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        
        # 使用模型进行预测
        predictions = model.predict(face)[0]
        max_index = np.argmax(predictions)
        emotion = expression_labels[max_index]
        
        # 在原图上绘制绿色矩形框标记人脸区域，并添加表情标签
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # 显示带有表情标注的结果图像窗口
    cv2.imshow('Emotion Detection', frame)
    
    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
camera.release()
cv2.destroyAllWindows()