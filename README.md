### Automated System for Calving Time Prediction and Cattle Classification Utilizing Trajectory Data and Movement Features

In this project, we used YOLOv8 for face detection.<br> 
Customized Tracking Algorithm (CTA) for tracking.<br> 
ResNet50 and SVM for Global IDs identification. <br> 
Comparison of three total movement features and statistical analysis are used for calving types of classification such as abnormal or normal cattle. <br> 
Comparison of three cumulative movement features and statistical analysis are used for calving time prediction for each cow. <br> 

### Project Description
Our system involved identifying and verifying a cow's identity including Local IDs and Global IDs from video frames.<br> 
The system integrated advanced tracking algorithms and movement analysis to predict calving times and classify cattle as normal or abnormal based on behavioral patterns.<br> 
Utilizing trajectory data, the system monitored cattle movement over a 12-hour period, enabling precise prediction of calving times well in advance.<br> 

### Cattle Face Detection (YOLOv8)
Our approach focused on accurately detecting cattle faces in various lighting and environmental conditions within calving pens by removing some noises such as person or trucks.<br>

### Cattle Tracking (CTA)
Our CTA ensured accurate and continuous tracking of individual cattle, even in scenarios involving occlusions or overlapping movements.<br>

### Cattle Global IDs Identification (ResNet50 with SVM)

The global IDs identification process ensured that each cattle's tracking and behavior analysis is consistent across multiple camera feeds, improving overall system accuracy. <br> 
The system’s ability to maintain global IDs over time allows for long-term monitoring and trend analysis of each individual cow's behavior.<br> 
An optimization logic within CTA aligns local track IDs with global IDs, enhancing reliability during long-term tracking and identification IDs.<br> 

### Calving Types Classification (Abnormal or Normal Classification)
Our system classified calving types into abnormal or normal categories by analyzing movement patterns using three key features: Total Euclidean Distance (TD), Total Magnitude of Acceleration (TA), and Total Moving Average of Triangle Area (TMA).

### Calving Time Prediction
Our system predicted calving time prediction for each cow by analyzing movement patterns using three key features: Cumulative Euclidean Distance (CD), Cumulative Magnitude of Acceleration (CA), and Cumulative Moving Average of Triangle Area (CMA).

## Table of Contents
- [System Diagram](#system-diagram)
- [Installation](#installation)
- [Authors](#authors)
- [License](#license)

## System Diagram
![System Diagram](https://github.com/WaiHninEaindrarMg/Automated-System-for-Calving-Time-Prediction-and-Cattle-Classification/blob/main/results/overview.png)

## Installation
1. Clone the repository:
```
git clone https://github.com/WaiHninEaindrarMg/Automated-System-for-Calving-Time-Prediction-and-Cattle-Classification.git
```

2. Install Ultralytics , check here for more information (https://github.com/ultralytics)
We trained our custom detection model and also added our model. If you want to use pretrained YOLO Model : (https://docs.ultralytics.com/tasks/segment/#how-do-i-load-and-validate-a-pretrained-yolo-segmentation-model)
```
pip install ultralytics
```

3. Install tensorflow:
```
pip install tensorflow==2.10.1
```

4. Install torch, torchvision, torchaudio:
```
pip install torch torchvision torchaudio
```

5. Install scikit-learn:
```
pip install scikit-learn
```

6. Install joblib:
```
pip install joblib
```

7. Install ipywidgets:
```
pip install ipywidgets
```

## Instruction
1. Run this file https://github.com/blob/main/detect_track_identify.py
```
python detect_track_identify.py
```
After running the specified file, the script automatically stores these folders and datasets:
![Testing Folders](https://github.com/WaiHninEaindrarMg/Automated-System-for-Calving-Time-Prediction-and-Cattle-Classification/blob/main/results/testing_folders.png)

This is testing result of images of integrating detection, tracking and identification.
![Result](https://github.com/WaiHninEaindrarMg/Automated-System-for-Calving-Time-Prediction-and-Cattle-Classification/blob/main/results/testing_results.gif)


2. Run this file calving_types_classification_and_calving_time_prediction.ipynb
```
Run calving_types_classification_and_calving_time_prediction.ipynb
```
In this calving_types_classification_and_calving_time_prediction.ipynb , There include the detailed experimental code for calving types classification and calving cattle classification. <br>
For the calving types of classification such as abnormal or normal, we used three features and made a comparison.<br>
1. Total Euclidean Distance (TD)<br>
2. Total Magnitude of Acceleration (TA)<br>
3. Total Moving Average of Triangle Area (TMA)<br>
After making comparisons, TD got the best accuracy. After that, TA and TMA followed. <br>
This result analysis plot for testing accuracy for all 20 cattle. Our system classified all the cattle as abnormal or normal.<br>
![Result](https://github.com/WaiHninEaindrarMg/Automated-System-for-Calving-Time-Prediction-and-Cattle-Classification/blob/main/results/cattle_classification.gif)<br>


For the calving time prediction, we used three features and made a comparison.<br>
1. Cumulative Euclidean Distance (CD)<br>
2. Cumulative Magnitude of Acceleration (CA)<br>
3. Cumulative Moving Average of Triangle Area (CMA)<br>
After making comparisons, CD predicted all 20 cattle of their calving times with a precision of ±6 hours,  CA predicted ± 9 hours,  CMA predicted ± 8  hours, respectively. <br>
In this, our system can also predict the precise calving hour for each of the cow by using these three features. <br>
This result analysis plot for testing accuracy for all 20 cattle. Our system classified all the cattle as abnormal or normal.<br>
![Result](https://github.com/WaiHninEaindrarMg/Automated-System-for-Calving-Time-Prediction-and-Cattle-Classification/blob/main/results/calving_time_prediction.gif)<br>


##
## Authors
👤 : Wai Hnin Eaindrar Mg  <br> 
📧 : [waihnineaindrarmg@gmail.com](mailto:nc22003@student.miyazaki-u.ac.jp) <br> 
👤 : Thi Thi Zin (corresponding-author)<br> 
📧 : [thithi@cc.miyazaki-u.ac.jp](mailto:thithi@cc.miyazaki-u.ac.jp) <br> 
👤 : Pyke Tin <br> 
👤 : Masaru Aikawa <br> 
👤 : Kazuyuki Honkawa  <br> 
👤 : Yoichiro Horii <br> 

## License
This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE.md file for details.

