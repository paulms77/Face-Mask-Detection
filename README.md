# Face-Mask-Detection-System
얼굴 마스크 착용 여부 시스템(Face Mask Detection System)

> 얼굴 마스크 착용 여부 시스템은 비디오 혹은 실시간 웹캠에서 마스크 착용 여부를 감지합니다.

## Dependencies
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [PyTorch](https://pytorch.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Face Recognition](https://github.com/ageitgey/face_recognition)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Python Image](https://pypi.org/project/pillow/)
- [NumPy](https://numpy.org/)


## Dataset
데이터셋은 두가지 클래스로 구성되어있습니다.
* __Without mask images__ : 마스크를 착용하지 않은 이미지 8718장 + 부적절하게 착용한 이미지 4723장
* __With mask images__ : 마스크를 착용한 이미지 4109장

(참고로 이미지 데이터셋은 Kaggle에서 수집하였습니다.)


## 얼굴 마스크 착용 여부 시스템
* ### 학습 데이터셋 구축

  * #### 1. 마스크 증강 과정
  
  <img width="500" src="https://github.com/user-attachments/assets/83946be6-c1a2-4c4e-adbc-cca0a48594dd">
  
  * ####
  
    * ##### ① 얼굴 인식 및 추출: 사람 이미지에서 face_recognition을 이용하여 얼굴 인식 및 추출
    
    * ##### ② 마스크 이미지 착용: 마스크를 착용하지 않은 얼굴 이미지에 마스크 이미지를 착용 (부적절한 착용 / 적절한 착용)

  * #### Result (결과):
    * ##### 마스크를 착용하지 않은 얼굴 이미지 6912장 추가로 생성
    * ##### 마스크를 부적절하게 착용한 얼굴 이미지 6270장 추가로 생성
    * ##### 마스크를 정상적으로 착용한 얼굴 이미지 6270장 추가로 생성
   

  * #### 참고로 마스크 이미지 착용은 균일하게 배분하였습니다. (흰색 마스크 착용 < 796 검은색 마스크 착용 < 1952 파란색 마스크 < 추출된 얼굴 이미지 수)

  * #### 이미지 증강 과정

    * ##### RandomRotation(무작위 회전)
      * ###### : 실제 이미지에는 얼굴이 항상 완벽한 각도로 정렬되어 있지 않을 수 있습니다. 따라서 이 기능을 사용하면 다른 각도로 착용한 얼굴 이미지가 생성되며, 정렬에 대한 오류를 더 잘 일반화하도록 합니다.
        
    * ##### RandomHorizontalFlip(랜덤 수평 뒤집기)
      * ###### : 실제로 비대칭 때문에 좌우 반전을 사용하는 경우가 있습니다. 따라서 좌우 반전이나 방향에 관계없이 일반화하도록 합니다.
     
    * ##### brightness(밝기)=0.2, contrast(대비)=0.2, saturation(채도)=0.2, hue(색조)=0.2
       * ###### : 실제로 실내 및 실외 환경, 조명이 좋지 않은 경우 또는 카메라 설정으로 촬용한 이미지와 같은 다양한 환경을 시뮬레이션합니다. 이는 조명과 색상 변화에 대한 견고성을 높여 이미지 밝기나 색상 변화의 차이에 관계없이 일반화하도록 합니다.

   * #### Result (결과):
  
   <img width="500" alt="image" src="https://github.com/user-attachments/assets/b9f6a6e0-ceb2-41f2-b350-bdac39d45129">

* ### 마스크 착용 여부 인식 시스템
  * #### 신경망 네트워크 구조
   * ##### MobileNetv2를 기반으로 Transfer Learning 전이 학습  커스텀 분류기를 추가했습니다.

  <img width="500" alt="image" src="https://github.com/user-attachments/assets/10a683fc-8ce9-423e-95b7-03d9211f6ef7">
  
* ### 마스크 착용 여부 검사 시스템
  * #### 얼굴 최소 인식률 30% 이상
    * ##### 실험 결과에 따르면 최소 인식률을 0.5(50%)로 하게 되면 외부 환경적 요인으로 사람 얼굴을 인식 못하는 경우가 종종 발생함 -> 0.3(30%)으로 설정하여 인식률을 높임.
      
  * #### 마스크 착용 확률 > 마스크 미착용 확률, 마스크 착용 확률 70% 이상 초록색 -> 마스크 착용
  * #### 마스크 착용 확률 < 마스크 미착용 확률, 마스크 미착용 확률 70% 이상 빨간색 -> 마스크 미착용

## 비디오(동영상)로 실행할 경우
```
video_path = './input.mp4'
vs = cv2.VideoCapture(video_path)
```

##  실시간 웹캠으로 실행할 경우
```
vs = cv2.VideoCapture(0)
```

## 얼굴 마스크 착용 여부 시스템 실행하기
```
$ python3 face-mask-detection.py
```


## 학습 결과
* ### plot 결과:
<img width="400" alt="image" src="https://github.com/user-attachments/assets/ab248597-6823-46f8-91c5-0cf49dc107ff">

* ### classification_report 결과:
<img width="400" alt="image" src="https://github.com/user-attachments/assets/46ec7974-b0bf-481a-b85e-7dbcbba23b56">

### ⚠️ 기존 시스템의 문제점 해결
기존의 모델은 다양한 색상의 마스크를 감지하거나 부적절한 마스크 착용(코 마스크, 턱 마스크)을 감지하는 데 어려움을 겪고 있습니다.

또한 환경적 요인에 따라 정확도가 급격히 떨어지는 경우가 존재합니다.

> 이를 해결하기 위해서 다양한 색상의 마스크 착용 이미지와 부적절한 마스크 착용 얼굴 이미지를 추가적으로 생성하고 이미지 증강 과정을 통해 개선하였습니다.


## License
[MIT LICENSE](https://github.com/paulms77/Face-Mask-Detection-System/blob/main/LICENSE)

