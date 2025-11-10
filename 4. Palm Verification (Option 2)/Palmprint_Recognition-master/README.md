# Palmprint Recognition System# Palmprint Recognition System# Palmprint_Recognition



A biometric authentication system using SIFT (Scale-Invariant Feature Transform) for palmprint recognition.This project is mainly to complete the palmprint feature extraction and classification tasks. The data set contains 99 people's palm print pictures, in which 3 palm print pictures of each person are distributed in the training set, and the other 3 palm print pictures are distributed in the test set. In this project, I tried the traditional method use SIFT to extract features and KNN for classification which get accuracy of 97.31%, and also tried the convolutional neural network method such as ResNet which get accuracy of 83.16%. In addition, I also tried to use the Gaussian filter, Gabor filter,etc. to process the palmprint image and extract the texture from the palmprint image, but these methods have not improved the accuracy of palmprint recognition.



## 🔑 FeaturesA biometric authentication system using SIFT (Scale-Invariant Feature Transform) and CNN (Convolutional Neural Networks) for palmprint recognition.



- **SIFT-based Authentication**: Multi-layer validation system with feature matching## 参考博客：

- **6-Layer Validation**: SIFT + Texture + Geometric + Template + SSIM + Edge detection

- **High Security**: 0.45% False Accept Rate (FAR)## 🔑 Features[【Pytorch】使用ResNet-50迁移学习进行图像分类训练](https://blog.csdn.net/heiheiya/article/details/103028543)

- **Dataset Support**: Compatible with Tongji and PolyU palmprint databases



## 📁 Project Structure

- **SIFT-based Authentication**: Multi-layer validation system with feature matching[【pytorch】数据增强](https://wizardforcel.gitbooks.io/learn-dl-with-pytorch-liaoxingyu/4.7.1.html)

```

├── SIFT_DIP.py                 # Main SIFT implementation- **CNN Implementation**: Deep learning approach using ResNet-18 with contrastive learning

├── texture_extraction_DIP.py   # Texture feature extraction

├── tongji_sift_auth.py         # SIFT authentication script- **6-Layer Validation**: SIFT + Texture + Geometric + Template + SSIM + Edge detection[opencv python SIFT（尺度不变特征变换）](https://segmentfault.com/a/1190000015709719)

├── tongji_dataset_prep.py      # Dataset preparation utilities

├── palm_auth_ultimate_fixed.py # Complete authentication pipeline- **Dataset Support**: Compatible with Tongji and PolyU palmprint databases

├── Palmprint/                  # Palmprint dataset

└── Tongji_Palmprint/           # Tongji dataset[OpenCV-Python教程:41.特征匹配](https://www.jianshu.com/p/ed57ee1056ab)

```

## 📁 Project Structure

## 🚀 Installation

[opencv python 特征匹配](https://segmentfault.com/a/1190000015735549)

### Prerequisites

- Python 3.8+```

- OpenCV

- NumPy├── SIFT_DIP.py                 # Main SIFT implementation[opencv中 cv2.KeyPoint和cv2.DMatch的理解](https://blog.csdn.net/qq_29023939/article/details/81130987)

- scikit-image

├── resnet18_DIP.py             # CNN ResNet-18 model

### Setup

```bash├── texture_extraction_DIP.py   # Texture feature extraction[K近邻算法](https://www.cnblogs.com/ybjourney/p/4702562.html)

# Install dependencies

pip install opencv-python numpy scikit-image pillow matplotlib├── tongji_sift_auth.py         # SIFT authentication script

```├── tongji_cnn_trainer.py       # CNN training script

├── tongji_dataset_prep.py      # Dataset preparation utilities

## 💻 Usage├── palm_auth_ultimate_fixed.py # Complete authentication pipeline

├── CNN_Partner/                # CNN partner implementation

### SIFT Authentication├── Palmprint/                  # Palmprint dataset

```python└── Tongji_Palmprint/           # Tongji dataset

python tongji_sift_auth.py```

```

## 🚀 Installation

### Complete Authentication Pipeline

```python### Prerequisites

python palm_auth_ultimate_fixed.py- Python 3.8+

```- OpenCV

- NumPy

### Dataset Preparation- PyTorch (for CNN)

```python- scikit-image

python tongji_dataset_prep.py

```### Setup

```bash

## 📊 Performance# Install dependencies

pip install opencv-python numpy torch torchvision scikit-image pillow matplotlib

### SIFT System Results```

- **Overall Accuracy**: 87.23%

- **GAR** (Genuine Accept Rate): 50.67%## 💻 Usage

- **FAR** (False Accept Rate): 0.45%

- **FRR** (False Reject Rate): 49.33%### SIFT Authentication

- **TRR** (True Reject Rate): 99.55%```python

- **Processing Time**: ~0.5-1.0 seconds per authenticationpython tongji_sift_auth.py

- **Test Dataset**: 595 comparisons across 50 individuals```



## 🎯 System Architecture### CNN Training

```python

### 6-Layer Validation Systempython tongji_cnn_trainer.py

```

1. **SIFT Feature Matching**

   - Extracts 1200 max features per image### Dataset Preparation

   - Uses Lowe's ratio test (threshold: 0.75)```python

   - Minimum 60 matches required for personal imagespython tongji_dataset_prep.py

   - Minimum 3 matches required for dataset authentication```



2. **Texture Analysis**## 📊 Performance

   - Calculates texture similarity using correlation

   - Threshold: 0.7### SIFT System

- **Accuracy**: 87.23%

3. **Geometric Verification**- **GAR** (Genuine Accept Rate): 50.67%

   - Validates spatial relationships between matched keypoints- **FAR** (False Accept Rate): 0.45%

   - Uses RANSAC for outlier removal- **Processing Time**: ~0.5-1.0 seconds



4. **Template Matching**### CNN System (ResNet-18)

   - Performs normalized cross-correlation- **Accuracy**: 91.68%

   - Threshold: 0.6- **GAR**: 79.42%

- **FAR**: 1.21%

5. **SSIM (Structural Similarity Index)**- **Processing Time**: ~0.12 seconds (GPU)

   - Measures perceptual similarity

   - Threshold: 0.5## 🎯 Key Components



6. **Edge Detection Verification**### SIFT Implementation

   - Compares edge patterns using Canny edge detection- Feature extraction with 1200 max features

   - Ensures structural consistency- Lowe's ratio test (threshold: 0.75)

- Multi-layer validation for robustness

## 🔬 Methodology

### CNN Implementation

### Preprocessing Pipeline- ResNet-18 architecture

- Bilateral filtering for noise reduction- 256-D embeddings

- CLAHE (Contrast Limited Adaptive Histogram Equalization)- NT-Xent contrastive loss

- Gamma correction for lighting normalization- Cosine similarity matching (threshold: 0.8)



### Feature Extraction## 📚 Dataset

- SIFT detector with configurable parameters

- Contrast threshold: 0.02Tested on:

- Edge threshold: 10- **Tongji Palmprint Database**: 50 individuals, 595 test comparisons

- Sigma: 1.6- **PolyU Palmprint Database**: Compatible



### Matching Strategy## 🤝 Contributing

- Brute-force matcher with L2 norm

- Lowe's ratio test for robust matchingThis project was developed as part of a Digital Image Processing course comparing traditional and deep learning approaches to palmprint recognition.

- Multi-layer validation for final decision

## 📄 License

## 📚 Dataset

This project is for educational purposes.

Tested on:
- **Tongji Palmprint Database**: 50 individuals, 595 test comparisons
- **PolyU Palmprint Database**: Compatible

## 🛡️ Security Analysis

The system achieves excellent security characteristics:
- **Very Low False Accept Rate (0.45%)**: Minimizes unauthorized access
- **High True Reject Rate (99.55%)**: Accurately rejects imposters
- **Trade-off**: Higher False Reject Rate (49.33%) may require multiple authentication attempts

## ⚠️ Limitations

- False Reject Rate of 49.33% means genuine users may need 2-3 attempts
- Processing time of 0.5-1.0 seconds is slower than modern deep learning approaches
- O(n) matching complexity affects scalability with large databases

## 🤝 Contributing

This project was developed as part of a Digital Image Processing course exploring traditional computer vision approaches to biometric authentication.

## 📄 License

This project is for educational purposes.
