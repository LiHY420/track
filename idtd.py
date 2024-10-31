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

def process_image(image):
    # 参数设置
    ro = 11
    ri = 10
    delta_b = newRingStrel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    result = MNWTH(gray, delta_b, bb)  # 检测运动

    # 使用自适应阈值来增强弱小目标的检测
    binaryImg = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # 形态学操作，清除噪声并填补小孔
    kernel = np.ones((3, 3), np.uint8)
    binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, kernel)
    binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, kernel)

    # 获取连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImg, connectivity=4)

    # 找到最大的连通区域的质心
    if num_labels > 1:
        # 过滤掉面积小于一定阈值的区域
        min_area_threshold = 50  # 可调整此值以检测较小目标
        valid_centroids = []

        for i in range(1, num_labels):  # 从1开始，跳过背景
            if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                valid_centroids.append(centroids[i])

        if valid_centroids:
            # 选择第一个有效质心（可以根据需要进行其他选择）
            centroid = valid_centroids[0]
            return int(centroid[0]), int(centroid[1])  # 返回质心坐标 (x, y)

    return None  # 如果没有检测到对象，返回 None

if __name__ == "__main__":
    # 示例用法
    image_path = r"I:\wll\images\15648.bmp"  # 替换为你的图像路径
    image = cv2.imread(image_path)

    if image is not None:
        centroid = process_image(image)
        if centroid:
            print(f"质心坐标: {centroid}")
        else:
            print("没有检测到对象。")
    else:
        print("无法读取图像。")
