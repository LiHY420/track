import cv2
import numpy as np
import time
import math

initialPoint = None
pointSelected = False

def my_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def show_img(frame, start, algorithm_name):
    end = time.time()
    ms_double = (end - start) * 1000
    fps = 1000 / ms_double if ms_double > 0 else 0
    print(f"it took {ms_double:.2f} ms")

    # 在图像上显示 FPS 和算法名称
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Algorithm: {algorithm_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 512)
    cv2.imshow("result", frame)
    cv2.waitKey(1)

# 手动实现 Otsu 阈值算法
def otsu_threshold(src):
    histogram, _ = np.histogram(src, bins=256, range=(0, 256))
    total_pixels = src.size
    sum_all = np.dot(np.arange(256), histogram)

    weight_bg, sum_bg = 0, 0
    max_variance, threshold = 0, 0

    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance_between > max_variance:
            max_variance = variance_between
            threshold = t

    print(f"Otsu threshold: {threshold}")
    return threshold

def process_image(image):
    # 转换为灰度图像并模糊处理
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.bilateralFilter(gray_frame, 9, 75, 75)  # 双边滤波平滑火光边缘

    # 计算Scharr梯度
    scharr_grad_x = cv2.Scharr(gray_frame, cv2.CV_16S, 1, 0)
    scharr_grad_y = cv2.Scharr(gray_frame, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(scharr_grad_x)
    abs_grad_y = cv2.convertScaleAbs(scharr_grad_y)
    scharr_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 计算Otsu阈值并进行二值化处理
    otsu_thresh = otsu_threshold(scharr_image)
    _, binary_img = cv2.threshold(scharr_image, otsu_thresh, 255, cv2.THRESH_BINARY)

    # 使用连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    min_area = 500  # 忽略面积小于500像素的连通域

    # 仅保留较大的连通域
    mask = np.zeros(binary_img.shape, dtype=np.uint8)
    for label in range(1, num_labels):  # 从1开始，跳过背景
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == label] = 255

    # 计算质心位置
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) > 0:
        weights = mask[y_indices, x_indices].astype(float)
        total_weight = np.sum(weights)
        total_weight_x = np.sum(x_indices * weights)
        total_weight_y = np.sum(y_indices * weights)

        # 计算加权平均位置
        center_x = total_weight_x / total_weight
        center_y = total_weight_y / total_weight

        return int(center_x), int(center_y)  # 返回质心坐标 (x, y)

    return None  # 如果没有检测到对象，返回 None

if __name__ == "__main__":
    # 示例用法
    image_path = r"I:\wll\images\15648.bmp"  # 替换为你的图像路径
    image = cv2.imread(image_path)

    if image is not None:
        start = time.time()
        centroid = process_image(image)
        if centroid:
            print(f"质心坐标: {centroid}")
            # 在图像上绘制质心
            cv2.circle(image, centroid, 10, (0, 255, 0), -1)  # 绘制绿色圆点
            show_img(image, start, "Image Processing")
        else:
            print("没有检测到对象。")
    else:
        print("无法读取图像。")
