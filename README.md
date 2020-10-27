# Real time mask-detection on Rasp 3B+ use Yolov3-tiny

## Model Selection
Mask –Detection을 하기 위해 모델을 선정하던중 네가지 후보 모델을 고려하였다.
- Faster R-CNN
 조사해본결과,R-cnn은 정확하고 겹쳐지거나 작은 사물에 대한 인식률은 높지만 많이 느리고 실시간을 생각하고 만든 네트워크는 아니므로 배제하였다.빠르고 사용이 쉽고 비교적 정확하지만 겹쳐진 사물의 구분이 어렵다는 단점이 있다.
 - Harr Classifier
Haar Classifier는 엄청 단순하고 빠르며 단순하다 하지만 영상의 밝기 값을 이용하기 때문에 조명,대조비에 영향을 많이 받고Sample의 양이나 질에 따라 성능에 영향을 크다 그리고 제일 큰 단점은 다소 정확하지 않다는 점이다.
- SSD
SSD(Single Shot MultiBox Detector)는 비교적 빠르고 정확하며,백본 네트워크를 바꿔가며 사용이 용이하여 성능과 속도를 어느정도 조절이 가능하며 유사네트워크가 많이 있지만,사용하기 쉽지가 않다는 단점이 있었다.
- YOLO
YOLO(You Only Look One)은 빠르고 비교적 정확하며,비교적 사용이 쉽다.하지만 겹쳐진 사물의 구분이 어렵다는 단점이 있다.

이 사항들을 모두 고려해봤을 때 yolo를 사용하는게 제일 적합하다고 생각하여 yolo를 사용하게 되었다.
## Data Preparation
https://github.com/tzutalin/labelImg(데이터 라벨링 툴)을 이용하여 2개의 class(mask,no-mask)를 yolo데이터셋을 만듬.각 클래스 별 100개정도 총 200개의 데이터로 모델 학습을 진행하였다.
validation과 train 데이터의 비율은 9:1로 진행하였다.처음에 소량의 데이터로 진행해서 그런지 detection이 제대로 되지않아 kaggle에서 데이터셋을 구한뒤 다시 진행 하였다.
https://www.kaggle.com/andrewmvd/face-mask-detection
하지만 해당 데이터셋은 pascal voc 형식의 xml데이터셋이었고,yolo 형식 xml 데이터셋으로 전환을 하기위해
https://bblib.net/entry/convert-voc-to-yolo-xml-to-yolo-xml-to-txt
해당 모듈을 사용하였다.
또한 마스크를 이상하게 착용한 것 까지 탐지 시켜주기 위해 클래스를 한 개 추가 시켜 총 3개의 클래스로 진행을 하였다.
windows 환경에서 compile을 수행하였고
모델은 real-time에 적합한 tiny-yolo를 사용하였다.
loss graph와 iteration별 validation set을 이용한 map 수치 그래프이다.

YOLO-tiny :About 2 hours YOLO :About 15 hours
GPU : gtx 1650..
GCP등을 활용했다면 더 빠른 시간안에 가능함

위에서 알 수 있듯이 MAP는 47%정도 나오는걸 알 수 있다.다음은 일반 yolo 모델 분석이다.(추후 추가 예정)

MAP=Mean Average Precision 클래스 ap의 전체 평균이다.
위 두 결과를 보고 알 수 있는 점
일반 모델과 tiny 모델과의 성능차이는 map측면에서 봤을 때 약 17%정도 된다.
3번째 class를 잘 detect하지 못하는 이유는 아무래도 데이터 부족이 원인이라고 추정된다.
77.13+44.59+18.66 / 3 
TP= 제대로 감지한 것
FP=감지해야하는데 못잡은것
FN=감지하면 안되는데 잡힌 것

IOU Intersection Over Union을 뜻함(교집합 /합집합)
Ground-truth :예측하고자 하는 관심영역을 뜻함
Precision :모델에서 예측된 관심영역


결과물

tiny 모델의 단일 image input 결과이다


맞춘것들은 비교적 잘 맞춘다.
동영상 결과는 내가 나중에 다 정리해서 드리겠음.
ppt에 약간 짤로 정리해서 써야 할 듯 ?
real time 결과물 
이것도 녹화해야 하는데 아직 못함 조만간 할 예정
raspberry에서 
tiny 모델의 경우 0.8초에 1프레임 일반 모델일 경우 15초에 1 프레임 분석 가능 
아쉬운 점 
Object Detection 모델에서 고려해야 하는건 세가지이다.
첫 번째,IOU를 고려하지 않고,ground truth를 잡았는지 ?
두 번쨰,ground truth를 잡았을 때 IOU가 어느정도 되는지 ?
세 번째,오탐지를 어느정도 하는지 ? 
1.여기서 알 수 있는 문제점 Iou가 50%로 위의 사진처럼 예측이 제대로 되는건 아니다.즉,잘 맞추긴 맞췄으나,맞춰야 하는 부분의 50%정도만 예측하고 있는 상황이다.

		YOLO					Yolo-tiny

둘 다 비교적 잘잡긴 하나 tiny모델은 오탐지 1건 미탐지 1건의 차이를 찾을 수 있다.

		YOLO-Tiny				YOLO
