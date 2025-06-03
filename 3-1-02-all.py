# 实验程序：Made by DDXie@43，实验日期：2025年5月
# 实验目的：用于学习计算机视觉中的“看到”图片中的人脸，并进一步“看清”人脸的五官特征，并标记出68个人脸关键点。

# 导入所需的库
import dlib  # Dlib：用于人脸检测与面部关键点定位
import cv2   # OpenCV：用于图像读取、显示和处理
import numpy as np  # NumPy：用于数组操作

def detect(frame):
    """
    定义一个函数 detect，用于检测图像中所有人脸，并标注68个面部关键特征点。
    
    参数:
        frame: 输入的一帧图像（BGR格式）
    """

    # 创建一个人脸检测器对象
    detector = dlib.get_frontal_face_detector()
    
    # 加载预训练的68个面部关键点检测模型文件
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # 将输入图像转换为灰度图
    # 因为Dlib的人脸检测和关键点预测通常在灰度图上进行
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用detector进行人脸检测
    # 第二个参数表示图像向上采样的次数，值越大能检测到更小的人脸，但计算量增加
    # 此处设置为1，即对图像放大一次后再检测，以提升多张人脸的识别率
    rects = detector(gray, 1)  # 返回的是 dlib.rectangle 类型的矩形列表
    
    # 遍历检测到的每一个人脸区域
    for rect in rects:
        # 获取人脸框的坐标信息
        x1, y1 = rect.left(), rect.top()     # 左上角坐标
        x2, y2 = rect.right(), rect.bottom() # 右下角坐标
        
        # 将坐标重新封装为 dlib.rectangle 对象
        dlib_rect = dlib.rectangle(x1, y1, x2, y2)
        
        # 在灰度图上使用 shape_predictor 预测该人脸的68个关键点
        shape = predictor(gray, dlib_rect)
        
        # 提取所有关键点的坐标并转换为 NumPy 数组，便于后续处理
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        # 遍历每一个关键点并在原图上绘制圆形标记
        for (x, y) in landmarks:
            # 在图像上画出绿色的小圆点（半径为1，颜色为绿色）
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

# 主程序开始

# 读取指定路径的图像文件
frame = cv2.imread("001image.jpg")

# 检查图像是否成功加载
if frame is None:
    raise FileNotFoundError("无法读取图像文件，请检查路径是否正确。")

# 调用 detect 函数，进行人脸检测与关键点标注
detect(frame)

# 显示标注后的图像窗口
cv2.imshow("LandMark All Faces", frame)

# 将处理完成的图像保存为新的文件
cv2.imwrite('001img_landmark_all_detected.jpg', frame)

# 等待任意按键按下后关闭所有OpenCV窗口
cv2.waitKey(0)