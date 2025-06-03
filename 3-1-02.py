# 实验程序：Made by DDXie@43，实验日期：2025年5月
# 实验目的：用于学习计算机视觉中的“看到”图片中的人脸，并进一步“看清”人脸的五官特征，并标记出68个人脸关键点。

# dlib 可能需要安装特定的系统版本，如：
# pip install dlib‑19.19‑cp38‑cp38‑win_amd64.whl
# 然而，windows系统可能还需要先安装相关的cmake依赖包。请参考其官网：
# https://cmake.org/download/

# 导入必要的库
import dlib  # Dlib库，用于面部特征点检测等高级功能
import numpy as np  # NumPy库，用于数值计算
import cv2  # OpenCV库，用于图像处理

def detect(frame):
    """
    定义一个函数detect，用于检测输入图像中的人脸，并标出68个面部关键特征点。
    参数:
        frame: 输入的一帧图像（BGR格式）
    """
    
    # 使用Dlib提供的正面人脸检测器
    detector = dlib.get_frontal_face_detector()
    
    # 加载预先训练好的用于预测68个人脸关键特征点的模型
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # shape_predictor_68_face_landmarks.dat是Dlib中68个人脸关键特征点的模型
    
    # 将图像转换为灰度图，因为Dlib人脸检测器在灰度图像上工作得更好
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测灰度图中的所有人脸，返回的是矩形列表，每个矩形对应一张人脸
    rects = detector(gray, 0)
    
    # 遍历所有检测到的人脸
    for i in range(len(rects)):
        # 对每张人脸使用Dlib的形状预测器来获取68个特征点的位置
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rects[i]).parts()])
        
        # 遍历所有的特征点并绘制圆圈标记
        for idx, point in enumerate(landmarks):
            # 获取每个特征点的坐标
            pos = (point[0, 0], point[0, 1])
            
            # 在原始图像上绘制圆圈，半径为1，颜色为绿色
            cv2.circle(frame, pos, 1, color=(0, 255, 0))

# 读取输入图像
frame = cv2.imread("001image.png")

# 调用detect函数进行人脸检测和面部特征点标注
detect(frame)

# 显示标注了面部特征点后的图像
cv2.imshow("face found and landmark", frame)

# 保存标注后的图像
cv2.imwrite('001img_landmark_detected.png', frame)

# 等待按键按下后关闭窗口
cv2.waitKey(0)