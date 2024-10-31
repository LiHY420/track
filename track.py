import cv2
import numpy as np
import time

def show_img(frame, start, algorithm_name):
    # 计算并显示处理时间和帧率
    end = time.time()
    ms_double = (end - start) * 1000
    fps = 1000 / ms_double if ms_double > 0 else 0
    print(f"处理时间: {ms_double:.2f} 毫秒")

    # 显示FPS和算法名称
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"算法: {algorithm_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("result", frame)
    cv2.waitKey(1)

def process_image(frame):
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 中心区域ROI裁剪
    roi_frame = gray_frame[gray_frame.shape[0] // 4: 3 * gray_frame.shape[0] // 4,
                           gray_frame.shape[1] // 4: 3 * gray_frame.shape[1] // 4]

    # 使用Scharr算子计算梯度图像
    scharr_grad_x = cv2.Scharr(roi_frame, cv2.CV_16S, 1, 0)
    scharr_grad_y = cv2.Scharr(roi_frame, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(scharr_grad_x)
    abs_grad_y = cv2.convertScaleAbs(scharr_grad_y)
    scharr_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 自适应Otsu阈值
    _, binary_img = cv2.threshold(scharr_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 获取连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    # 过滤小连通域，保留面积大于指定阈值的区域
    min_area = 1000  # 小区域过滤阈值
    large_components = [(i, stats[i], centroids[i]) for i in range(1, num_labels) if stats[i][cv2.CC_STAT_AREA] > min_area]

    # 初始化质心变量
    final_centroid = None

    # 检查是否存在覆盖目标的高亮大面积区域
    for i, stat, centroid in large_components:
        # 检查连通域是否为大面积高亮区域并覆盖目标
        if stat[cv2.CC_STAT_WIDTH] > roi_frame.shape[1] * 0.5 or stat[cv2.CC_STAT_HEIGHT] > roi_frame.shape[0] * 0.5:
            # 瘦身操作（腐蚀）以减小连通域面积
            mask = (labels == i).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            eroded_mask = cv2.erode(mask, kernel)

            # 重新计算瘦身后区域的质心
            moments = cv2.moments(eroded_mask)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"]) + frame.shape[1] // 4
                center_y = int(moments["m01"] / moments["m00"]) + frame.shape[0] // 4
                final_centroid = (center_x, center_y)
        else:
            # 正常情况下使用计算得到的质心
            center_x, center_y = int(centroid[0]) + frame.shape[1] // 4, int(centroid[1]) + frame.shape[0] // 4
            final_centroid = (center_x, center_y)

    return final_centroid

def main():
    video_path = r"D:\dolphin_dataset\处理后\原始的\track-train-1\video.mp4"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"无法打开视频: {video_path}")
        return -1

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取帧数

    try:
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                print("读取视频帧失败！")
                break

            start = time.time()  # 开始计时

            centroid = process_image(frame)  # 处理图像，获取质心坐标
            if centroid:
                x, y = centroid
                cv2.rectangle(frame, (max(x - 30, 0), max(y - 30, 0)),
                              (min(x + 30, frame.shape[1]), min(y + 30, frame.shape[0])),
                              (0, 0, 255), 2)
                print(f"质心位置：({x:.2f}, {y:.2f})")

            # 显示效果图
            show_img(frame, start, algorithm_name="sot")

    finally:
        # 确保在任何情况下都能释放资源
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
