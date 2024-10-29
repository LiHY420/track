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

# 手动实现 Otsu 阈值算法，包含提前终止条件
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

        # 提前终止条件
        if weight_bg > total_pixels * 0.9975 or weight_fg > total_pixels * 0.9975:
            break

    print(f"Otsu threshold: {threshold}")
    return threshold

# 鼠标事件回调函数：选择跟踪目标
def on_mouse(event, x, y, flags, param):
    global initialPoint, pointSelected
    if event == cv2.EVENT_LBUTTONDOWN:
        initialPoint = (x, y)
        pointSelected = True

def main():
    video_path = r"D:/dolphin_dataset/lhy/output_video.avi"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Failed to open video: {video_path}")
        return -1

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取帧数
    FPS = video.get(cv2.CAP_PROP_FPS)  # 获取FPS
    lightFlag = True
    num = 0

    cv2.namedWindow("00", 0)
    cv2.resizeWindow("00", 640, 512)

    try:
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                print("Failed to read frame!")
                break

            cv2.imshow("00", frame)

            # 转换为灰度图像并模糊处理
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.bilateralFilter(gray_frame, 9, 75, 75)  # 双边滤波平滑火光边缘

            # 腐蚀操作
            b2 = np.ones((4, 4), np.uint8)
            gray_frame = cv2.erode(gray_frame, b2)

            start = time.time()
            print(f"min= {num // 20 // 60}, sec= {num // 20 % 60}")
            num += 1

            if not lightFlag:
                gray_frame = 255 - gray_frame

            # 裁剪ROI区域
            roi_frame = gray_frame[gray_frame.shape[0] // 6: 5 * gray_frame.shape[0] // 6,
                                   gray_frame.shape[1] // 6: 5 * gray_frame.shape[1] // 6]

            # 计算Scharr梯度
            scharr_grad_x = cv2.Scharr(roi_frame, cv2.CV_16S, 1, 0)
            scharr_grad_y = cv2.Scharr(roi_frame, cv2.CV_16S, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(scharr_grad_x)
            abs_grad_y = cv2.convertScaleAbs(scharr_grad_y)
            scharr_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            cv2.namedWindow("11", 0)
            cv2.resizeWindow("11", 640, 512)
            cv2.imshow("11", scharr_image)

            # 计算Otsu阈值并进行二值化处理
            otsu_thresh = otsu_threshold(scharr_image)
            _, binary_img = cv2.threshold(scharr_image, otsu_thresh, 255, cv2.THRESH_BINARY)

            cv2.namedWindow("binaryImg", 0)
            cv2.resizeWindow("binaryImg", 640, 512)
            cv2.imshow("binaryImg", binary_img)

            # 使用连通域分析来忽略分散的连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
            min_area = 500  # 忽略面积小于500像素的连通域

            # 仅保留较大的连通域
            mask = np.zeros(binary_img.shape, dtype=np.uint8)
            for label in range(1, num_labels):  # 从1开始，跳过背景
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    mask[labels == label] = 255

            # 计算质心位置，使用新的掩码图像
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) > 0:
                weights = mask[y_indices, x_indices].astype(float)
                total_weight = np.sum(weights)
                total_weight_x = np.sum(x_indices * weights)
                total_weight_y = np.sum(y_indices * weights)

                # 计算加权平均位置
                center_x = total_weight_x / total_weight + frame.shape[1] / 6
                center_y = total_weight_y / total_weight + frame.shape[0] / 6

                cv2.rectangle(frame,
                              (max(int(center_x - 30), 0), max(int(center_y - 30), 0)),
                              (min(int(center_x + 30), frame.shape[1]), min(int(center_y + 30), frame.shape[0])),
                              (0, 0, 255), 2)
                print(f"全图质心位置：({center_x:.2f}, {center_y:.2f})")

            # 显示效果图窗口
            algorithm_name = "sot"
            show_img(frame, start, algorithm_name)

    finally:
        # 确保在任何情况下都能释放资源
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
