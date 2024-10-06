# Face-Mask-Detection-System
얼굴 마스크 착용 여부 시스템(Face Mask Detection System)

얼굴 마스크 착용 여부 시스템은 비디오 혹은 실시간 웹캠에서 마스크 착용 여부를 감지합니다.

## Dependencies
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [PyTorch](https://pytorch.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Face Recognition](https://github.com/ageitgey/face_recognition)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Python Image]([https://pypi.org/project/pillow/](https://github.com/python-pillow/Pillow))
- [NumPy](https://numpy.org/)

## Dataset
데이터셋은 두가지 클래스로 구성되어있습니다.
* __Without mask images__ : 마스크를 착용하지 않은 이미지 장 + 부적절하게 착용한 이미지 장
* __With mask images__ : 마스크를 착용한 이미지 장

(참고로 이미지 데이터셋은 Kaggle에서 수집했습니다.)

## 얼굴 마스크 착용 여부 시스템
* ### 학습 데이터셋 구축
  마스크 증강
  tranform 증강

* ### 마스크 착용 여부 인식 시스템
  딥러닝 모델

* ### 마스크 착용 여부 검사 시스템
  검사 처리

## 얼굴 마스크 착용 여부 시스템 실행하기
파일.ipynb 코드

## 결과
plot
classification_report
