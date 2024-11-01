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
    ro = 10  # 减小外半径
    ri = 11  # 减小内半径
    delta_b = newRingStrel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    result = MNWTH(gray, delta_b, bb)  # 检测运动

    # 使用自适应阈值进行二值化
    binaryImg = cv2.adaptiveThreshold(result, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       11,  # 块大小
                                       2)    # 常数C

    # 获取连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImg, connectivity=4)

    # 找到最大连通区域的质心
    if num_labels > 1:
        valid_centroids = []
        valid_areas = []

        # 遍历所有连通组件，选择小于500像素的目标
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= 500:  # 仅考虑小于等于500像素的目标
                valid_centroids.append(centroids[i])
                valid_areas.append(area)

        # 如果有符合条件的连通域，找到面积最大的一个
        if valid_areas:
            max_index = np.argmax(valid_areas)
            centroid = valid_centroids[max_index]  # 获取质心坐标
            return int(centroid[0]), int(centroid[1])  # 返回质心坐标 (x, y)

    return None  # 如果没有检测到对象，返回 None

if __name__ == "__main__":
    # 示例用法
    image_path = r"C:\vision\gd11-akesai\video\004812\17967.png"  # 替换为你的图像路径
    image = cv2.imread(image_path)

    if image is not None:
        centroid = process_image(image)
        if centroid:
            x, y = centroid
            box_size = 20  # 边长为20像素
            top_left = (x - box_size // 2, y - box_size // 2)
            bottom_right = (x + box_size // 2, y + box_size // 2)

            # 绘制边长20像素的正方形框
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            print(f"质心坐标: {centroid}")

            # 显示图像
            cv2.imshow("Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("没有检测到符合条件的对象。")
    else:
        print("无法读取图像。")
