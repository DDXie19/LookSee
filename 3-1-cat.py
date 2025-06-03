# 实验程序：Made by DDXie@43，实验日期：2025年5月
# 实验目的：用于学习计算机视觉中的“看到”图片中的猫脸，并框出猫脸。

# 导入需要用到的库
from PIL import Image  # PIL 是 Python Imaging Library，用于图像处理（虽然本程序中未直接使用）
import cv2             # OpenCV 库，用于图像处理和猫脸检测
import numpy           # 用于处理数组和矩阵运算

def detect(frame):
    """
    检测输入图像中的正面猫脸，并在检测到的猫脸周围绘制矩形框。
    
    参数:
        frame: 输入的一帧图像（BGR格式，由OpenCV读取）
    """

    # 加载预训练的Haar级联分类器文件，用于检测正面猫脸
    # 注意：这里的"frontalface.xml"需要是你自己下载并重命名的haar级联文件
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")

    # 将图像从BGR颜色空间转换为灰度图
    # 猫脸检测通常在灰度图像上进行，这样更高效且效果不受颜色影响
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用detectMultiScale方法检测图像中的猫脸
    # scaleFactor: 图像缩放比例，用于补偿远距离猫脸尺寸较小的问题
    # minNeighbors: 检测框保留阈值，数值越大检测越严格，越少误检
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)

    # 输出检测到的猫脸数量
    print("Found {0} faces!".format(len(faces)))

    # 遍历所有检测到的猫脸，在原图上绘制绿色矩形框标记猫脸位置
    for (x, y, w, h) in faces:
        # frame: 要绘制的图像
        # (x, y): 矩形左上角坐标
        # (x + w, y + h): 矩形右下角坐标
        # (0, 255, 0): 绿色边框（BGR格式）
        # 2: 线条粗细
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# 主程序部分开始

# 读取一张图片文件 "01image.png"
# imread函数返回一个NumPy数组，表示图像数据（BGR格式）
frame = cv2.imread("cat37.jpg")

# 调用detect函数进行猫脸检测，会修改frame的内容（添加猫脸框）
detect(frame)

# 显示检测结果图像
# "face found" 是窗口标题
cv2.imshow("cat face found", frame)

# 将检测后的图像保存为新文件 "01image_detected.png"
cv2.imwrite("cat37_face_detected.jpg", frame)

# 等待用户按键（参数0表示无限等待），以便查看显示的图像
# 按任意键后关闭所有OpenCV创建的窗口
cv2.waitKey(0)