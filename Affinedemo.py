import time
import os
from PIL import Image

import cv2
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 将车牌进行旋转
def rotation(in_img, rect_size, center, angle):
    # 0度则不旋转
    # 检查图像是否是灰度图，如果是，则添加一个维度以模拟颜色通道

    if in_img.ndim == 2:
        in_img = in_img[:, :, np.newaxis]

    # 创建一个更大的图像，尺寸为原图的2倍
    in_large = np.zeros((int(in_img.shape[0] * 3.2), int(in_img.shape[1] * 3.2), in_img.shape[2]), dtype=in_img.dtype)

    x = max(in_large.shape[1] / 2 - center[0], 0)
    y = max(in_large.shape[0] / 2 - center[1], 0)

    width = min(x + in_img.shape[1], in_large.shape[1]) - x
    height = min(y + in_img.shape[0], in_large.shape[0]) - y

    if width != in_img.shape[1] or height != in_img.shape[0]:
        return False, None

    in_large[int(y):int(y+height), int(x):int(x+width)] = in_img


    new_center = (in_large.shape[1] / 2, in_large.shape[0] / 2)

    rot_mat = cv2.getRotationMatrix2D(new_center, angle, 1.0)
    mat_rotated = cv2.warpAffine(in_large, rot_mat, (in_large.shape[1], in_large.shape[0]), flags=cv2.INTER_CUBIC)

    # 如果原始图像是灰度图，从旋转后的图像中裁剪出的区域也应该是灰度图
    if in_img.shape[2] == 1:
        img_crop = cv2.getRectSubPix(mat_rotated, (int(in_large.shape[0]), int(rect_size[1])*2), new_center)[:, :, np.newaxis]
    else:
        img_crop = cv2.getRectSubPix(mat_rotated, (int(in_large.shape[0]), int(rect_size[1])*2), new_center)

    return True, img_crop.squeeze()  # 使用squeeze()移除单维度条目

# 对车牌实施放缩
def fangsuo(src, a, b):
    rows, cols, channel = src.shape
    M = np.float32([[a, 0, 0], [0, b, 0]])
    dst = cv2.warpAffine(src, M=M, dsize=(cols, rows))
    return dst
# 计算斜率
def is_deflection(in_img, angle,direction):
    assert in_img.ndim == 2  # 确保图像是单通道的
    # 获取了图像的宽高数
    nRows, nCols = in_img.shape
    # 将行一分为三
    comp_index = [nRows // 4, nRows // 2, 3 * nRows // 4]
    lengths = []

    for i in range(3):
        index = comp_index[i]
        # 这行代码从图像in_img中获取索引为index的行。
        row = in_img[index, :]
        j = 0
        for element in row:
            if element==0:
                j+=1
        #     跳出来了
        lengths.append(j)

    maxlen = max(lengths[2], lengths[0])
    minlen = min(lengths[2], lengths[0])

    PI = 3.14159265
    g = np.tan(angle * PI / 180.0)
    slope_pre=''
    if maxlen - lengths[1] > nCols / 32 or lengths[1] - minlen > nCols / 32:
        slope_can_1 = abs(lengths[2] - lengths[0]) / (comp_index[2]-comp_index[0])
        print(slope_can_1)
        slope_can_2 = abs(lengths[1] - lengths[0]) / (comp_index[1]-comp_index[0])
        print(slope_can_2)
        # 而代码的目的是找到一个与这个目标值最接近的斜率
        slope_pre = 'slope_can_1' if abs(slope_can_1 - g) <= abs(slope_can_2 - g) else "slope_can_2"
        if direction=='left':
            if slope_pre=='slope_can_1':
                slope = (slope_can_1*9+slope_can_2)/10
                # slope =slope_can_1
            else:
                # slope = (slope_can_1  + slope_can_2*3) / 4
                slope = (slope_can_1+slope_can_2*9)/10

        else:
            if slope_pre == 'slope_can_1':
                 slope = -(slope_can_1 * 9 + slope_can_2) / 10
            #     slope = -slope_can_1
            else:
                 slope = -(slope_can_1 + slope_can_2 * 9) / 10
                # slope = -slope_can_2
        print("斜率为"+str(slope))
        return True, slope
    else:
        slope = 0

    return False, slope
# 根据斜率进行仿射变换
def affine_transform(in_img, slope):
    """
    根据斜率执行仿射变换。
    """
    height, width = in_img.shape[:2]
    xiff = np.abs(slope) * height

    if slope > 0:
        plTri = np.array([[0, 0], [width - xiff - 1, 0], [xiff, height - 1]], dtype=np.float32)
        dstTri = np.array([[xiff / 2, 0], [width - 1 - xiff / 2, 0], [xiff / 2, height - 1]], dtype=np.float32)
    else:
        plTri = np.array([[xiff, 0], [width - 1, 0], [0, height - 1]], dtype=np.float32)
        dstTri = np.array([[xiff / 2, 0], [width - 1 - xiff + xiff / 2, 0], [xiff / 2, height - 1]], dtype=np.float32)

    warp_mat = cv2.getAffineTransform(plTri, dstTri)
    affine_mat = cv2.warpAffine(in_img, warp_mat, (width, height), flags=cv2.INTER_LINEAR)

    return affine_mat
# 进行高斯滤波
def bi_fillter(img):

    time1 = time.time()
    dst1 = cv2.bilateralFilter(src=img, d=0, sigmaColor=15, sigmaSpace=20)
    cv2.imwrite("bi_img.jpg", dst1)  # 图片大小100*50 效果更好。
    # plt.imshow(dst1)
    # plt.show()
    time2 = time.time()
    print("gray_bi",time2-time1)
    return dst1

def process_img(image_path, angle,color,ddate,height_plate):
    # 读取图像
    if not os.path.exists("./rectified/"+ddate):
        os.makedirs("./rectified/"+ddate)
    if angle<0:
        direction= 'right'
    else:
        direction ='left'
    in_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bi_fillter(in_img)
    if in_img is None:
        print("Error loading image")
        return
    # 假定中心点和需要的矩形大小
    center = (in_img.shape[1] / 2, in_img.shape[0] / 2)
    # 裁剪的尺寸不确定
    rect_size = (in_img.shape[1], height_plate+10)  # 你希望裁剪的尺寸
    # 执行旋转,0度则不予以旋转
    if angle == 180 or angle ==0:
        img_rotated= in_img
    else:
        success, img_rotated = rotation(in_img, rect_size, center, angle)
        if success:
            # 显示或保存结果
            # cv2.imshow('Rotated Image', img_rotated)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            output_path = "./rotation_results/" + image_path
            cv2.imwrite(output_path, img_rotated)
        else:
            print("Rotation failed.")
    # 应用二值化
    # 根据颜色设置阈值，白色车牌为200，黄色车牌为100,蓝色车牌为115，绿牌为100
    if color =='white':
        thresh = 200
    elif color=='yellow' or color == 'green':
        thresh =100
    elif color == 'blue':
        thresh = 115
    else:
        thresh = 0
        print('缺少color属性值!')
    # 设置最大像素值
    maxValue = 255
    # 应用阈值化
    _, image_binary = cv2.threshold(img_rotated, thresh, maxValue, cv2.THRESH_BINARY)

    # 显示二值化图像
    # cv2.imshow('Binary Image', image_binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    is_deflected, slope = is_deflection(image_binary, angle,direction)  # 假设我们想检测10度的倾斜
    # 测试一下倾斜矫正的效果
    # img_rotated = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    # img_rotated = cv2.imread('test.png')  # 替换为你的图片路径
    if is_deflected:
        # 二值化图像进行仿射变换的结果
        out_img = affine_transform(image_binary, slope)
        # 显示或保存结果
        # cv2.imshow('Affine Transformation', out_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 执行仿射变换
        out_img = affine_transform(img_rotated, slope)
        # 显示或保存结果
        # cv2.imshow('Affine Transformation', out_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("./rectified/" + ddate + "-"+str(angle)+"-ocr.jpg", out_img)
    else:
        print("No significant deflection detected.")
        cv2.imwrite("./rectified/" + ddate + "-" + str(angle) + "-ocr.jpg", img_rotated)
    ocr_path="./rectified/" + ddate + "-" + str(angle) + "-ocr.jpg"
    return ocr_path

# 可以忽视
# if __name__ == "__main__":
#     src = 'demo.png'
#     # 旋转角度不确定
#     angle = -30  # 旋转角度
#     process_img(src,angle)
