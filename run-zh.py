import cv2
import numpy as np

from anime_face_detector import create_detector

# 创建检测器
detector = create_detector('yolov3')

# 读取图像
image = cv2.imread('assets/test3.jpg')

# 进行人脸检测
preds = detector(image)

# 遍历所有检测结果
for i, pred in enumerate(preds):
    # 画边界框
    bbox = pred['bbox'].astype(int)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(image, f'Box {i+1}', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 画并标注关键点
    keypoints = pred['keypoints'].astype(int)
    for j, kp in enumerate(keypoints):
        cv2.circle(image, (kp[0], kp[1]), 3, (0, 0, 255), -1)
        cv2.putText(image, str(j+1), (kp[0], kp[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 保存标注后的图像
cv2.imwrite('assets/test3-out.jpg', image)
