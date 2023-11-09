# NUS EE5907 Pattern Recognition - CA2 Face Recognition

This is a course project submitted to EE5907. The aim of this project is to construct a face recognition system via Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), and Convolution Neural Network (CNN). PCA and LDA are used to perform data dimensionality reduction and visualization to understand the underlying data distribution. Then lower-dimensional data are classified based on the nearest-neighbor classifier. In addition, SVM and CNN are also used to classify face images.

As for the detailed project requirements, you can find the corresponding file [CA2_Requirements.pdf](https://github.com/Jonathan3016/EE5907_Pattern-Recognition/blob/88e2e0af0d379e9f7bdd475ba13e35fbda4cb41c/Assignment%202/files/CA2_Requirements.pdf). The details of this project were written in the report, which can be found at [EE5907_CA2_Report.pdf
](https://github.com/Jonathan3016/EE5907_Pattern-Recognition/blob/88e2e0af0d379e9f7bdd475ba13e35fbda4cb41c/Assignment%202/files/EE5907_CA2_Report.pdf).

## Data

This project is conducted on the [CMU PIE](https://www.ri.cmu.edu/publications/the-cmu-pose-illumination-and-expression-database/) dataset and selfie photos taken by myself which stored in a folder called `Selfie`. There are in total 68 different subjects and I randomly selected 25 subjects from it. For each chosen subject, use 70% of the images provided for training and use the remaining 30% for testing. In addition to the provided CMU PIE images, I took 10 selfie photos for myself, converted them to grayscale images, resized them into the same resolution as the CMU PIE images (i.e., 32×32) and split them into 7 for training and 3 for testing. 

* The dataset processing code used in this project is stored in the ["data"](https://github.com/Jonathan3016/EE5907_Pattern-Recognition/tree/29e2c622cd6c6b4b66e6bb91ffb60a35f9b0e073/Assignment%202/data) folder along with the processing results in the format '.npy'.
  
```
├── data
│   ├── dataset.py
│   ├── train_x.npy
│   ├── train_y.npy
│   ├── test_x.npy
│   ├── test_y.npy
```

## Prerequisites

All the packages needed for this project are documented in the ["requirements.txt"](https://github.com/Jonathan3016/EE5907_Pattern-Recognition/blob/29e2c622cd6c6b4b66e6bb91ffb60a35f9b0e073/Assignment%202/requirements.txt) file, along with their exact versions.

* All the dependencies can be installed using the following command:

```sh
  pip install -r requirements.txt
```

Some important dependencies and their version numbers are listed below：

- Python 3.11
- numpy == 1.26.0
- matplotlib == 3.7.1
- libsvm-official == 3.32.0
- torch == 2.1.0 + cu118
- pandas == 1.5.3

## Getting Started

This project contains a total of 4 main components: Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA),  Support Vector Machine (SVM), and Convolution Neural Network (CNN).

### Principal Component Analysis (PCA)

The files related to PCA are stored in three separate folders as follows:

```
├── model
│   ├── PCA.py
├── run
│   ├── run_PCA.py
├── result
│   ├── pca
│   ├──   ├── PCA_2-dimensions.png
│   ├──   ├── PCA_3-dimensions.png
│   ├──   ├── PCA_eigenfaces.png
│   ├──   ├── pca_accuracy.csv
```

* The model build for PCA is stored in 'PCA.py', to run the PCA model, you can open this project and use the following command:

```sh
  python run\run_PCA.py
```

### Linear Discriminant Analysis (LDA)

The files related to LDA are stored in three separate folders as follows:

```
├── model
│   ├── LDA.py
├── run
│   ├── run_LDA.py
├── result
│   ├── lda
│   ├──   ├── LDA_2-dimensions.png
│   ├──   ├── LDA_3-dimensions.png
│   ├──   ├── lda_500_samples_accuracy.csv
│   ├──   ├── lda_all_samples_accuracy.csv
```

* The model build for LDA is stored in 'LDA.py', to run the LDA model, you can open this project and use the following command:

```sh
  python run\run_LDA.py
```

### Support Vector Machine (SVM)

This project applies to `libsvm`, which you can download from [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and check out how to use. The files related to SVM are stored in three separate folders as follows:

```
├── model
│   ├── SVM.py
├── run
│   ├── run_SVM.py
├── result
│   ├── svm
│   ├──   ├── svm_accuracy.csv
```

* The model build for SVM is stored in 'SVM.py', to run the SVM model, you can open this project and use the following command:

```sh
  python run\run_SVM.py
```

### Convolution Neural Network (CNN)

This project applies to `torch`, which you can download from [here](https://pytorch.org/) and check out how to use. The files related to CNN are stored in three separate folders as follows:

```
├── model
│   ├── CNN.py
├── run
│   ├── run_CNN.py
├── result
│   ├── cnn
│   ├──   ├── best_model.pth
│   ├──   ├── CNN_test_accuracy.npy
│   ├──   ├── CNN_train_losses.npy
│   ├──   ├── CNN_train_loss.png
│   ├──   ├── CNN_test_accuracy.png
```

* The model build for CNN is stored in 'CNN.py', to run the CNN model, you can open this project and use the following command:

```sh
  python run\run_CNN.py
```
