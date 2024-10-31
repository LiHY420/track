import cv2
import numpy as np


def newRingStrel(ro, ri):
    d = 2 * ro + 1
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0  # 设置内部区域为零
    return se


def MNWTH(img, delta_b, bb):
    img_d = cv2.dilate(img, delta_b)  # 执行膨胀
    img_e = cv2.erode(img_d, bb)  # 执行腐蚀
    out = cv2.subtract(img, img_e)  # 从原始图像减去腐蚀后的图像
    out[out < 0] = 0  # 将负值设置为零
    return out


def process_image(image, prev_frame=None):
    # 参数设置
    ro = 5  # 适当缩小卷积核半径
    ri = 4
    delta_b = newRingStrel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 如果存在前一帧，进行帧差分，锁定运动区域
    if prev_frame is not None:
        gray_diff = cv2.absdiff(gray, prev_frame)
        _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)  # 二值化运动区域
    else:
        motion_mask = np.ones_like(gray, dtype=np.uint8) * 255  # 首帧设为全区域

    # 使用与周围像素的亮度差异来检测小目标
    local_mean = cv2.blur(gray, (3, 3))  # 计算局部平均亮度（小卷积核）
    diff = cv2.absdiff(gray, local_mean)  # 计算亮度差异图
    _, binaryImg = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)  # 亮度差异二值化

    # 应用运动掩码，仅保留运动区域
    binaryImg = cv2.bitwise_and(binaryImg, motion_mask)

    # 形态学开操作以去除噪声小区域
    kernel = np.ones((3, 3), np.uint8)
    binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, kernel)

    # 获取连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImg, connectivity=4)

    # 找到最大的连通区域的质心
    if num_labels > 1:
        min_area_threshold = 50  # 最小目标面积阈值
        valid_centroids = []

        for i in range(1, num_labels):  # 从1开始，跳过背景
            if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                valid_centroids.append(centroids[i])

        # 选择第一个有效质心（可根据具体需求更改）
        if valid_centroids:
            centroid = valid_centroids[0]
            return int(centroid[0]), int(centroid[1]), gray  # 返回质心坐标 (x, y) 及当前帧灰度图

    return None, gray  # 如果没有检测到对象，返回 None 和当前帧灰度图


if __name__ == "__main__":
    # 示例用法
    image_path = r"I:\wll\images\15648.bmp"  # 替换为你的图像路径
    image = cv2.imread(image_path)
    prev_frame = None  # 用于存储前一帧

    if image is not None:
        centroid, prev_frame = process_image(image, prev_frame)  # 传入前一帧
        if centroid:
            print(f"质心坐标: {centroid}")
        else:
            print("没有检测到对象。")
    else:
        print("无法读取图像。")
