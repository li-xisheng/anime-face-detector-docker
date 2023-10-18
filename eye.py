import cv2
import numpy as np
import os

from anime_face_detector import create_detector

# 创建检测器
detector = create_detector('yolov3')

# 定义输入文件路径和名称
#############################################
input_filepath = 'assets/test1.jpg'
#############################################
input_filename = os.path.basename(input_filepath)
filename_without_extension = os.path.splitext(input_filename)[0]

# 读取图像
image = cv2.imread(input_filepath)

# 进行人脸检测
preds = detector(image)

# 创建assets文件夹下，filename_without_extension为名的存储目录
os.makedirs(f'assets/{filename_without_extension}', exist_ok=True)

# 遍历所有检测结果
for idx, pred in enumerate(preds):
    # 画边界框
    bbox = pred['bbox'].astype(int)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(image, f'Box {idx+1}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 画并标注关键点
    keypoints = pred['keypoints'].astype(int)

    # 截取眼部图像并保存
    for eye_indices, eye_name in [([11, 12, 13, 14, 15, 16], 'eye1'), ([17, 18, 19, 20, 21, 22], 'eye2')]:
        eye_keypoints = keypoints[eye_indices, :2]
        x_min, y_min = np.min(eye_keypoints, axis=0) - 10  # 外拓 10px
        x_max, y_max = np.max(eye_keypoints, axis=0) + 10  # 外拓 10px
        eye_crop = image[y_min:y_max, x_min:x_max]
        # 存储到assets文件夹下，filename_without_extension为名的存储目录中
        cv2.imwrite(f'assets/{filename_without_extension}/{idx+1}-{eye_name}.jpg', eye_crop)
    
cv2.imwrite(f'assets/{filename_without_extension}/{filename_without_extension}-out.jpg', image)