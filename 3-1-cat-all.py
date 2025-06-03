# 实验程序：Made by DDXie@43，实验日期：2025年5月
# 实验目的：用于学习计算机视觉中的“看到”图片中的猫脸，并框出所有的猫脸。

# 导入必要的库
import cv2  # OpenCV库，用于图像和视频处理
import numpy as np  # NumPy库，主要用于数组操作（虽然本程序未直接使用，但通常配合OpenCV使用）

def detect(frame):
    """
    定义一个函数detect，用于检测输入图像中的猫脸。
    
    参数:
        frame: 输入的一帧图像（BGR格式）
    """

    # 加载预训练的猫脸检测分类器（Haar级联分类器）
    # 使用的是OpenCV自带的 haarcascade_frontalcatface_extended.xml 文件
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml'
    )
    
    # 检查分类器是否成功加载
    if face_cascade.empty():
        raise IOError("无法加载猫脸分类器文件，请确认文件路径！")

    # 将图像转换为灰度图
    # 因为OpenCV中的Haar级联分类器只能在灰度图像中进行人脸/猫脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用detectMultiScale方法检测猫脸
    # 参数说明：
    # - scaleFactor：图像缩放比例，用于补偿不同距离摄像机的猫脸大小
    # - minNeighbors：保留检测框的最小邻居数，值越小越敏感，但也可能更多误检
    # - minSize：最小检测窗口大小，小于该尺寸的区域不会被检测
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.04,      # 图像缩小比例，数值越小检测越细致，速度也更慢
        minNeighbors=3,        # 检测框保留阈值，数值越高检测结果越可靠，但可能漏检
        minSize=(30, 30),      # 最小检测对象尺寸
        flags=cv2.CASCADE_SCALE_IMAGE  # 标志位，表示正常缩放图像进行检测
    )
    
    # 输出检测到的猫脸数量
    print(f"检测到 {len(faces)} 张猫脸")
    
    # 遍历所有检测到的猫脸位置信息，并在原图上绘制矩形框
    for (x, y, w, h) in faces:
        # 绘制绿色矩形框，线宽为2像素
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 可选：在矩形框上方添加文字标签“Cat”
        cv2.putText(
            frame, 
            'Cat',                     # 要显示的文字内容
            (x, y-10),                 # 文字左下角坐标（在矩形框上方留出空间）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
            0.9,                       # 字体大小
            (0, 255, 0),               # 文字颜色（绿色）
            2                          # 文字粗细
        )

# 主程序开始

# 读取图像文件 "cat31.jpg"
image = cv2.imread("cat31.jpg")

# 检查图像是否成功读取
if image is None:
    raise FileNotFoundError("无法读取图像文件，请检查路径！")

# 调用detect函数进行猫脸检测
# 注意：传入的image是引用传递，函数内部修改会影响原图
detect(image)

# 将标记后的图像保存为新文件
cv2.imwrite("cat31_face_detected.jpg", image)

# 显示检测结果图像
cv2.imshow("Cat Faces Detected", image)

# 等待任意按键按下后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()