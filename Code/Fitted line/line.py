import cv2
import numpy as np
import pandas as pd
import os

# 创建一个空的DataFrame用于保存结果
results_df = pd.DataFrame(columns=['Image', 'Slope', 'Intercept'])

# 图像文件夹路径
image_folder = ''

# 遍历图像文件夹中的所有图片
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # 读取图像并预处理
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 提取肌肉线条的轮廓
        red_mask = cv2.inRange(image, (0, 0, 100), (50, 50, 255))

        _, contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # 拟合直线
        rows, cols = gray_image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.001, 0.001)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        line_slope = vy / vx
        line_intercept = y - line_slope * x

        # 在图像上绘制直线
        line_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        cv2.line(line_image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

        # 保存结果到DataFrame
        results_df = results_df.append({
            'Image': filename,
            'Slope': line_slope.item(),
            'Intercept': line_intercept.item()
        }, ignore_index=True)

        # 保存带有直线的图像
        output_image_path = os.path.join(image_folder, f'output_{filename}')
        cv2.imwrite(output_image_path, line_image)

# 将结果保存到Excel文件
excel_path = ''

results_df.to_excel(excel_path, index=False)