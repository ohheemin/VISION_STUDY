import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread('/home/ohheemin/Downloads/for_calibration.jpg') #이미지 path 설정
rows, cols, ch = img.shape
# 변환 점 설정
pts1 = np.float32([[0, 0], [640, 0], [0, 640]])

# 확대/축소 변환
pts2_scaling = np.float32([[0, 0], [480, 0], [0, 480]])  # 약 0.75배 축소
matrix_scaling = cv2.getAffineTransform(pts1, pts2_scaling)
result_scaling = cv2.warpAffine(img, matrix_scaling, (cols, rows))

# 회전 변환
angle = 45  # 45도 회전
matrix_rotation = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
result_rotation = cv2.warpAffine(img, matrix_rotation, (cols, rows))

# 전단 변환
pts2_shearing = np.float32([[0, 0], [540, 100], [0, 540]])  # x축 방향으로 약간의 전단 변환

matrix_shearing = cv2.getAffineTransform(pts1, pts2_shearing)
result_shearing = cv2.warpAffine(img, matrix_shearing, (cols, rows))

# 이동 변환
pts2_translation = np.float32([[100, 100], [740, 100], [100, 740]])  # 오른쪽 아래로 100픽셀 이동
matrix_translation = cv2.getAffineTransform(pts1, pts2_translation)
result_translation = cv2.warpAffine(img, matrix_translation, (cols, rows))

# 결과 이미지 그리기
plt.figure(figsize=(10, 10))

plt.subplot(221), plt.imshow(result_scaling), plt.title('Scaling')
plt.subplot(222), plt.imshow(result_rotation), plt.title('Rotation')
plt.subplot(223), plt.imshow(result_shearing), plt.title('Shearing')
plt.subplot(224), plt.imshow(result_translation), plt.title('Translation')

plt.show()