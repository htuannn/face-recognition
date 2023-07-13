# Measure Similarity-Based Facial Recognition


## Introduction
This repository contains the implementation of a facial recognition system based on similarity measurement. We focuses on measuring the similarity between facial features and uses a similarity-based approach to recognize faces.

## Implementation

### Face Detection Model
In this repository, we utilize the YOLOv5 model that has been pre-trained specifically for the task of face detection. YOLOv5 is a state-of-the-art object detection algorithm known for its accuracy and efficiency. I already put the `yolov5s.pt` inside.

### Siamese Model and Similarity Measure
The Siamese Model is responsible for embedding facial images into feature vectors. It takes an input image and converts it into a feature vector A. The system then matches this feature vector with a pre-embedded database of facial feature vectors to find the feature vector B with the smallest angular difference from A. The output of this recognition process is the corresponding identity label for the feature vector B.

To measure the similarity between feature vectors, we employ the cosine similarity. The cosine similarity can be calculated using the formula:

![Cosine similarity](https://raw.githubusercontent.com/sagarmk/Cosine-similarity-from-scratch-on-webpages/master/images/cos.png)

In this equation, `A` and `B` are the feature vectors being compared. The cosine similarity value ranges from -1 to 1, where 1 indicates perfect similarity and -1 indicates complete dissimilarity.

### Loss Function
We utilize the MagFace loss function, which is based on angular-margin-based classification. The MagFace loss function is described in the following paper: 

+ **MagFace Loss**: [Paper](https://arxiv.org/abs/2103.06627) [Code](https://github.com/IrvingMeng/MagFace) (MagFaceHeader: <img src="https://render.githubusercontent.com/render/math?math=\cos(\theta %2B f_m(x))">)

## Usage
### Prepare enviroment and image-data 
To get started, follow these steps:

1. Clone this repository by running the following command:
```
git clone https://github.com/htuannn/face-recognition
cd face-recognition
```

2. Install all the necessary packages by running:
```
pip install -r requirements.txt
```

3. Ensuring your image-database is set up according to the following structure:

```
face-recognition/
│
├── data_recognition/
│   ├── raw
│   │   ├── PersonA
|   │   │     ├img1.jpg 
|   │   │     └...
│   │   ├── PersonB
|   │   │     ├img1.jpg  
|   │   │     └...
|   │   └...     
|   │ 
│   ├──preprocessed
│   │   ├── PersonA
|   │   │     ├img1_cropped.jpg
|   │   │     └...
|   │   └...    
```

_Note-1:`face-recognition/data_recognition/raw` is the place to store the entire database of each person's raw images. Each sub-folder here corresponds to one identity, with the name of the sub-folder being the person's name._

_Note-2: `face-recognition/data_recognition/preprocessed` is the place to store the entire database of each person's facial cropped images. The storage rules here are the same as above._

### Data preprocessing to crop face from original image
With the images that you have collected, which maybe a photo of the whole person, now we will cut out the face separately. To do this, run the following command:

```
python align_data.py
```

The processed images will be stored in the following path: `data_recognition/preprocessed`

### Convert image to embedding vector
To convert the preprocessed face images into embedding vectors, run the following command:

```
python face_embedding/embedding_feat.py --backbone iresnet50 --resume weights\models\magface_iresnet50_MS1MV2_dp.pth --folderdataset_dir data_recognition/preprocessed --feat_list  data_recognition/preprocessed/face_embdding.txt
```

### Real-time recognition inference
For real-time face recognition inference using a webcam input, run the following command:

`python rec_cam.py --backbone iresnet50 --resume weights\models\magface_iresnet50_MS1MV2_dp.pth --view-img`

## Citation
1) [MagFace: A universal representation for face recognition and quality assessment](https://github.com/IrvingMeng/MagFace)
2) [yolov5](https://github.com/ultralytics/yolov5)  
