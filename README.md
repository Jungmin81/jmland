# Real time mask-detection on Rasp 3B+ use Yolov3-tiny


## About Object Detection

- Object Detection의 성능이란?

    Object detection 관련 논문을 읽다 보면 초기의 논문들은 대부분 성능에 `정확도` 지표를 사용하고 있는 것을 확인할 수 있습니다. Object Detection 뿐만 아니라 다양한 Task의 논문들을 살펴보면 대부분 연구 초기에는 주로 `정확도`라는 지표를 올리기 위한 연구를 수행합니다. Object Detection에서는 이 `정확도` 라는 지표를 어떻게 나타낼 수 있을까요?

    정확도의 계산은 주로 `정답(Ground Truth, 이하 GT)`와 모델이 예측한 결과`(Prediction)` 간의 비교를 통해 이루어집니다. Image Classification의 경우에는 GT가 이미지의 class인 반면, Object Detection은 `이미지의 각 object의 해당하는 Bounding Box와 Box 안의 class`를 의미합니다. 즉 정확도가 높다는 것은 모델이 GT와 유사한 Bounding Box를 예측(Regression)하면서 동시에 Box 안의 object의 class를 잘 예측(Classification)하는 것을 의미합니다. `즉 class도 정확하게 예측하면서, 동시에 object의 영역까지 잘 예측을 해야 합니다`.

    보통 Object Detection 논문에서 사용하는 정확도의 경우 Class를 예측하지 못하면 실패로 간주됩니다. Class를 올바르게 예측하였을 때의 Bounding Box의 정확도를 기준으로 정확도를 측정하게 됩니다. 이제 이 정확도를 어떻게 측정하는지에 대해 설명을 드리겠습니다.

- IoU (Intersection Over Union)

    ![precision_recall_iou](https://hsto.org/files/ca8/866/d76/ca8866d76fb840228940dbf442a7f06a.jpg)
    Object Detection에서 Bounding Box를 얼마나 잘 예측하였는지는 IoU라는 지표를 통해 측정하게 됩니다. `IoU(Intersection Over Union)`는 Object Detection, Segmentation 등에서 자주 사용되며, 영어 뜻 자체로 이해를 하면 “교집합/합집합” 이라는 뜻을 가지고 있습니다. 실제로 계산도 그러한 방식으로 이루어집니다. Object Detection의 경우 `모델이 예측한 결과와 GT, 두 Box 간의 교집합과 합집합을 통해 IoU를 측정`합니다.

    > 출처 : https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-how-to-measure-performance-of-object-detection/ 
## Model Selection

Mask –Detection을 하기 위해 모델을 선정하던중 네가지 후보 모델을 고려하였다.

- Faster R-CNN

     R-CNN은 정확하고 겹쳐지거나 작은 사물에 대한 인식률은 높지만 많이 느리고 `실시간을 생각하고 만든 네트워크는 아니므로` 배제하였다.
 
 - YOLO

    YOLO(You Only Look One)은 빠르고 비교적 정확하고 사용이 쉽다.하지만 `겹쳐진 사물의 구분이 어렵다`는 단점이 있다.

     ![image](https://user-images.githubusercontent.com/39875941/97379913-4a7c6b00-1909-11eb-81db-2878be4f3754.png) ![image](https://user-images.githubusercontent.com/39875941/97380452-67fe0480-190a-11eb-9e14-8e01632ee280.png)
 

- SSD



    SSD(Single Shot MultiBox Detector)는 비교적 빠르고 정확하며,백본 네트워크를 바꿔가며 사용이 용이하여 성능과 속도를 어느정도 조절이 가능하며 유사네트워크가 많이 있지만,`사용하기 쉽지가 않다`는 단점이 있었다.

    ![image](https://user-images.githubusercontent.com/39875941/97380911-5537ff80-190b-11eb-9d42-4dccb5853a6e.png)

- Haar Classifier

    Haar Classifier는 엄청 단순하고 빠르며 단순하다 하지만 영상의 밝기 값을 이용하기 때문에 조명,대조비에 영향을 많이 받고,Sample의 양이나 질에 따라 성능에 영향을 크다 그리고 제일 큰 단점은 `다소 정확하지 않다는 점`이다.

    ![image](https://user-images.githubusercontent.com/39875941/97381191-e60edb00-190b-11eb-9d96-7da59feb1564.png)

    > 출처 : https://jetsonaicar.tistory.com/12 

위 사항들을 모두 고려해봤을 때 **YOLO**를 사용하는게 제일 적합하다고 생각하여 **YOLO**를 사용하게 되었다.

## Data Preparation


![image](https://user-images.githubusercontent.com/39875941/97381302-2b330d00-190c-11eb-8e09-94678695f098.png)

Labelimg를 이용하여 2개의 class(mask,no-mask)를 yolo데이터셋을 만듬.각 클래스 별 100개정도 총 200개의 데이터로 모델 학습을 진행하였다.

> https://github.com/tzutalin/labelImg(데이터 라벨링 툴)

validation과 train 데이터의 비율은 9:1로 진행하였다.처음에 소량의 데이터로 진행해서 그런지 detection이 제대로 되지않아 kaggle에서 데이터셋을 구한뒤 다시 진행 하였다.
```python
test_train_split.py

file_train = open(r'C:\Users\JM\Desktop\JM\mask-detection\train_test\train.txt', 'w')
file_test = open(r'C:\Users\JM\Desktop\JM\mask-detection\train_test\test.txt', 'w')
```
split을 수행하기전 파일의 경로와 확장자를 바꿔줘야 한다.

> https://www.kaggle.com/andrewmvd/face-mask-detection(마스크 데이터셋)

하지만 해당 데이터셋은 pascal voc 형식의 xml데이터셋이었고,yolo 형식 xml 데이터셋으로 전환을 하기위해 해당 모듈을 사용하였다.

> https://bblib.net/entry/convert-voc-to-yolo-xml-to-yolo-xml-to-txt(PASCAL VOC to YOLO converter)





```python
voc2yolo_converter.py

dirs = ['annotation']#경로 설정
classes = ['with_mask','without_mask','mask_weared_incorrect']#클래스 설정
~~~
    full_dir_path = cwd + '/' + dir_path #output path 지정
    output_path = full_dir_path +'/yolo/'
```
해당 파일에서 경로와 클래스를 적절하게 지정해줘야 한다.


해당 데이터셋은 마스크를 이상하게 쓴 데이터를 포함해 총 3개의 클래스로 데이터셋 준비를 하였다.

## Complie Yolo & train Model


![image](https://user-images.githubusercontent.com/39875941/97408437-6c90e000-193f-11eb-8986-26135692a27b.png)




> https://github.com/AlexeyAB/darknet

Windows 환경에서 Compile을 수행하였고 GPU 는 `GTX 1650`을 사용하였다.

Complie 관련 issue는 위 사이트를 참고했고,`Default Configure`을 사용하였다.
```ini
~~~.data
classes= 80
train  = <replace with your path>/trainvalno5k.txt
valid = <replace with your path>/testdev2017.txt
names = data/coco.names
backup = backup
eval=coco
```
.data파일의 경로와 클래스수를 사용환경에 맞게 수정해준다.그 외에도 .cfg파일을 수정해야 하는데 위의 사이트를 참고해볼것.

모델은 real-time에 적합한 tiny-yolo를 사용하였다.

다음 사진은 loss graph와 iteration별 validation set을 이용한 map 수치 그래프이다.단순 비교를 위해 일반 YOLO 모델도 분석해봤다.

| Yolo-tiny | Taken Time | Yolo | Taken Time |
|:---:|:---:|:---:|:---:|
|![chart_n_yolov3-tiny2](https://user-images.githubusercontent.com/39875941/97408940-3011b400-1940-11eb-87a8-b340f38c0deb.png)|**2 hours**|![chart_yolov3-custom](https://user-images.githubusercontent.com/39875941/97408945-31db7780-1940-11eb-94a1-b118bbfa7389.png)|**15 hours**|


GCP등을 활용했다면 더 빠른 시간안에 가능하다.

위에서 알 수 있듯이 tiny 모델의 best MAP는 `48%`정도 일반 모델의 best MAP는 `80%`정도  나오는걸 알 수 있다.


![image](https://user-images.githubusercontent.com/39875941/97413516-3a36b100-1946-11eb-83ad-75cee367c3e4.png)
![image](https://user-images.githubusercontent.com/39875941/97413760-8550c400-1946-11eb-955a-c020cb3a2f89.png)



위 두 결과를 보고 알 수 있는 점
일반 모델과 tiny 모델과의 성능차이는 MAP측면에서 봤을 때 약 `32%`정도 된다.

모든 class의 precision이 올라갔다는걸 알 수 있다.

```
MAP=Mean Average Precision 클래스별 ap의 전체 평균이다.
TP= 제대로 감지한 것
FP= 감지해야하는데 못잡은것
FN= 감지하면 안되는데 잡힌 것
IOU = Intersection Over Union을 뜻함(교집합 /합집합)
Ground-Truth(GT) = 예측하고자 하는 관심영역을 뜻함
Precision = 모델에서 예측된 관심영역
```



## Raspberry pi 3B+ Setting & files



Detector 모듈과 Live Detection모듈은 밑의 링크에서 참고했다.


> 출처 : https://github.com/rushad7/mask-detection 

```python
yolo-live-cv2.py

labelsPath = os.path.sep.join([args["yolo"], "custom.names"])
#class list file 
weightsPath = os.path.sep.join([args["yolo"], "yolov3-custom_final.weights"])
#weight file
configPath = os.path.sep.join([args["yolo"], "yolov3-custom.cfg"])#config file

~~~
    frame = cv2.resize(frame, dsize=(400,400), interpolation=cv2.INTER_CUBIC)#창 사이즈 설정
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),swapRB=True, crop=False)#train한 width와 height를 맞춰줘야함


```

위와 같이 경로 설정 및 기타 설정을 해줘야 한다.

custom-detector의 사용법은 py파일을 참고할 것.

## Real Time Detection result


```linux
python yolo-live-cv2.py --yolo yolo
```

![image](https://user-images.githubusercontent.com/39875941/97463455-1ba2db00-1983-11eb-9cc6-60d747ca5a70.png)

모든 설정을 끝마치치고 라즈베리파이 터미널에 해당 코드를 작성하면 다음과 같이 출력이 될 것이다.



![image](https://user-images.githubusercontent.com/39875941/97462038-aedb1100-1981-11eb-8131-b42588f40a25.png)
![image](https://user-images.githubusercontent.com/39875941/97462047-b0a4d480-1981-11eb-942a-f7b07ed1ecfb.png)
![image](https://user-images.githubusercontent.com/39875941/97462052-b26e9800-1981-11eb-96dc-6e086a7f0d5d.png)

pycam에 나오는 결과이다.성공적으로 세개의 클래스에 대해 분류가 가능하단걸 알 수 있다.


## Conclusion

| 비교 | Yolo | Yolo-tiny |
|:---:|:---:|:---:|
|Train time| 15 hours|2 hours|
|best MAP|80%|48%|
|IOU|69%|57%|
|fps|0.06fps|1.25fps|


성능면에서는 일반 Yolo 모델이 월등히 우수하지만 Real-Time에서 사용하기에는 Yolo-tiny모델이 더 사용하기 적합하다는 것을 알 수 있다.

| Case | Yolo | Yolo-tiny |
|:---:|:---:|:---:|
|1|![image](https://user-images.githubusercontent.com/39875941/97409490-0907b200-1941-11eb-8dc0-b3484ea1c4b4.png)|![image](https://user-images.githubusercontent.com/39875941/97409502-0e64fc80-1941-11eb-97da-34769381eac5.png)|
|2|![image](https://user-images.githubusercontent.com/39875941/97409515-145add80-1941-11eb-8eaf-44307ecfbf27.png)|![image](https://user-images.githubusercontent.com/39875941/97409508-12911a00-1941-11eb-9cf5-fbc4c391dc9f.png)|

Validation data의 test결과이다.자세히 확인해보면 일반 tiny 모델이 detection측면에서 좀 더 좋긴 하지만 눈으로 봐서는 별 차이가 없다는 것을 알 수 있다.

## 아쉬운 점 

Object Detection 모델에서 고려해야 하는건 세가지이다.

*첫 번째,IOU를 고려하지 않고,ground truth를 잡았는지 ?*

*두 번쨰,ground truth를 잡았을 때 IOU가 어느정도 되는지 ?*

*세 번째,오탐지를 어느정도 하는지 ?* 

- 여기서 알 수 있는 문제점 Iou가 50%로 위의 사진처럼 예측이 제대로 되는건 아니다.
즉,잘 맞추긴 맞췄으나,맞춰야 하는 부분의 50%정도만 예측하고 있는 상황이다.
`IOU 수치가 많이 낮은 상황이다.`



## 앞으로 적용해볼 것들

- Object Tracking을 위한 DeepSORT 알고리즘 적용

    ![image](https://ssvar.ch/wp-content/uploads/2019/12/object-detection-and-tracking-using-mediapipe.gif)
    ![sort](Result/deepsort.gif)
    현재의 detection은 단순히 모든 프레임을 탐지하는것 뿐이다.좌측의 detection을 수행하고 있다고 생각하면 된다.

    tracking알고리즘이 추가된 우측의 detection이 더 좋은 detection이라고 할 수 있다. 

- Kalman Filter

    칼만은 기존 추적하던 물체의 속도를 반영해서 다음 상황을 예측한다고 생각하면 쉽다.

    칼만 필터는 베이지안 추정과 같이 직접확률을 계산할 수 없는 경우 관련 된 값을 이용하여 원래 값을 구하는 것으로 predict <-> update 사이클로 이루어져 있다.

    ![image](https://user-images.githubusercontent.com/39875941/97794915-4622bb80-1c43-11eb-9741-607633fcdc89.png)


    1.과거의 값을 이용하여 현재값을 예측하고
    
    2.예측값과 측정값에 각각 노이즈를 반영한 뒤, 실제값을 예측한다.
    
    3.이 실제값을 다시 다음 측정에 사용한다.

    측정값을 그냥 쓰면 되는거 아니냐고 생각할 수 있지만 노이즈라는 개념이 들어가면 (원래 센서퓨전에 쓰려고 만든 알고리즘이므로) 측정값도 100% 신뢰할 수 없다는 것을 알 수 있다.

    또 칼만필터는 기본적으로 가우시안 분포로 값이 분포되어 있다고 가정하고 있으며(즉, 예측 값은 평균과 분산으로 표현될 수 있다.), 측정값의 분포가 가우시안이 아닐 경우에는 해당 분포에 맞는 변형된 칼만 알고리즘을 사용해야 한다.
- DeepSORT

    딥소트는 칼만필터 기본으로 딥러닝 피쳐(Re-Id)를 추가로 반영하여 헝가리안알고리즘을 수행한다고 말해도 크게 틀리지 않을 것이다.

    딥러닝 피쳐는 칼만필터의 한계 때문에 도입된 것으로 생각되는데, 칼만필터는 이전 속도를 기반으로 예측하기 때문에 실제로 SORT 나 칼만만 써서 트래킹하다보면 둘이 겹치는 (occlusion) 씬에서 에러가 많다.

    둘이 겹치는 부분에서 갑자기 서로 반대로 간다거나, 한명이 멈춰 있다가 나타난다거나 하는 경우가 생기면(실제로 자주 발생하는 경우이다.), 트래킹 아이디를 반대로 바꿔 버린다거나 (1<->2) 새로운 트래킹 아이디를 부여한다거나 (2->3) 하는 일이 생긴다.

- 정리 

    *SORT = 디텍터 + 칼만필터 + 헝가리안 알고리즘*

    *DeepSORT = 딥러닝 + SORT*

딥소트는 위와 같은 느낌으로, 실제로 코드를 보면 각 코드의 영역이 무엇을 하는지 파악하는 것이 크게 어렵지 않다. 개인적으로 딥소트의 트래킹 성능에 가장 크게 영향을 끼치는 것은 디텍션으로, 디텍션을 맡고 있는 YOLO3 의 성능이 가장 중요한 것 같다.

> 출처 : http://blog.haandol.com/2020/02/27/deep-sort-with-mxnet-yolo3.html 
