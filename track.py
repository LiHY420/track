import cv2
import numpy as np
import time


# 自定义函数：手动实现 Otsu 阈值算法
def otsu_threshold(src):
    histogram, _ = np.histogram(src, bins=256, range=(0, 256))
    total = src.size
    sumB = 0
    wB = 0
    sum1 = np.dot(np.arange(256), histogram)

    max_var = 0
    threshold = 0

    for t in range(256):
        wB += histogram[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF == 0:
            break

        sumB += t * histogram[t]

        mB = sumB / wB
        mF = (sum1 - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold


# 打开视频文件
video_path = 'C:/Users/12631/Downloads/J20240711fire/j190728.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("无法打开视频文件")
    exit()

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

output_file_path = "C:/Users/12631/Downloads/J20240711fire/2-j103500.txt"
roi_file_path = "C:/Users/12631/Downloads/J20240711fire/roi2-j103500.txt"

# 打开文件以保存质心和 ROI 数据
with open(output_file_path, 'w') as output_file, open(roi_file_path, 'w') as roi_file:
    num = 0
    total_processing_time = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 显示原始图像
        cv2.imshow("Original Frame", frame)

        # 转换为灰度图像并模糊处理
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

        # 腐蚀操作
        kernel = np.ones((4, 4), np.uint8)
        eroded_frame = cv2.erode(gray_frame, kernel)

        # 记录处理开始时间
        start_time = time.time()

        # 提取 ROI 区域
        roi_frame = eroded_frame[frame.shape[0] // 4:frame.shape[0] * 3 // 4,
                    frame.shape[1] // 4:frame.shape[1] * 3 // 4]

        # 计算 Scharr 梯度
        scharr_x = cv2.Scharr(roi_frame, cv2.CV_16S, 1, 0)
        scharr_y = cv2.Scharr(roi_frame, cv2.CV_16S, 0, 1)
        scharr_abs_x = cv2.convertScaleAbs(scharr_x)
        scharr_abs_y = cv2.convertScaleAbs(scharr_y)

        # 合并梯度图像
        scharr_image = cv2.addWeighted(scharr_abs_x, 0.5, scharr_abs_y, 0.5, 0)

        # 计算 Otsu 阈值
        otsu_thresh = otsu_threshold(scharr_image)
        _, binary_img = cv2.threshold(scharr_image, otsu_thresh, 255, cv2.THRESH_BINARY)

        # 计算二值化图像的质心
        moments = cv2.moments(binary_img)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv2.rectangle(frame, (center_x - 30, center_y - 30), (center_x + 30, center_y + 30), (0, 0, 255), 2)

            # 写入质心和 ROI 数据
            output_file.write(f"({center_x + frame.shape[1] // 4}, {center_y + frame.shape[0] // 4})\n")
            roi_file.write(f"({center_x - 30 + frame.shape[1] // 4}, {center_y - 30 + frame.shape[0] // 4}), "
                           f"({center_x + 30 + frame.shape[1] // 4}, {center_y + 30 + frame.shape[0] // 4})\n")

        # 计算处理时间
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # 转换为毫秒
        total_processing_time += processing_time

        print(f"Processing time: {processing_time:.2f} ms")
        num += 1

        # 显示处理后的图像
        cv2.imshow("Processed Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            break

# 计算并显示平均帧率
average_fps = num / (total_processing_time / 1000.0)
print(f"Average FPS: {average_fps:.2f}")

video.release()
cv2.destroyAllWindows()
