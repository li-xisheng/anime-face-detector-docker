import cv2
import numpy as np

from anime_face_detector import create_detector

# ディテクターの作成
detector = create_detector('yolov3')

# 画像の読み込み
image = cv2.imread('assets/test1.jpg')

# 顔の検出を行う
preds = detector(image)

# すべての検出結果をループ処理
for pred in preds:
    # バウンディングボックスを描画
    bbox = pred['bbox'].astype(int)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # キーポイントを描画
    keypoints = pred['keypoints'].astype(int)
    for kp in keypoints:
        cv2.circle(image, (kp[0], kp[1]), 3, (0, 0, 255), -1)

# 注釈付きの画像を保存
cv2.imwrite('assets/test1-out.jpg', image)
