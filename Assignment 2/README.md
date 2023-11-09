# NUS EE5907 Pattern Recognition - CA2 Face Recognition

This is a course project submitted to EE5907. The aim of this project is to construct a face recognition system via Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), and Convolution Neural Network (CNN). PCA and LDA are used to perform data dimensionality reduction and visualization to understand the underlying data distribution. Then lower-dimensional data are classified based on the nearest-neighbor classifier. In addition, SVM and CNN are also used to classify face images.

As for the detailed project requirements, you can find the corresponding file [CA2_Requirements.pdf](https://github.com/Jonathan3016/EE5907_Pattern-Recognition/blob/88e2e0af0d379e9f7bdd475ba13e35fbda4cb41c/Assignment%202/files/CA2_Requirements.pdf). The details of this project were written in the report, which can be found at [EE5907_CA2_Report.pdf
](https://github.com/Jonathan3016/EE5907_Pattern-Recognition/blob/88e2e0af0d379e9f7bdd475ba13e35fbda4cb41c/Assignment%202/files/EE5907_CA2_Report.pdf).

## Data

This project is conducted on the [CMU PIE](https://www.ri.cmu.edu/publications/the-cmu-pose-illumination-and-expression-database/) dataset and selfie photos taken by myself which stored in a folder called `Selfie`. There are in total 68 different subjects and I randomly selected 25 subjects from it. For each chosen subject, use 70% of the images provided for training and use the remaining 30% for testing. In addition to the provided CMU PIE images, I took 10 selfie photos for myself, converted them to grayscale images, resized them into the same resolution as the CMU PIE images (i.e., 32×32) and split them into 7 for training and 3 for testing. 

* The dataset processing code used in this project is stored in the data folder along with the processing results in the format '.npy'.
  
```
├── data
│   ├── dataset.py
│   ├── train_x.npy
│   ├── train_y.npy
│   ├── test_x.npy
│   ├── test_y.npy
```

## Prerequisites

