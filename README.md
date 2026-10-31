# Image-Based-PM2.5-Prediction
Internship at ISI,KOLKATA
# Image-Based PM2.5 Prediction

### Internship Project — Indian Statistical Institute (ISI), Kolkata

This repository contains the work carried out during my internship at the **Indian Statistical Institute (ISI), Kolkata)**, focusing on **image processing, computer vision, and machine learning for PM2.5 prediction**.

The project explored the relationship between atmospheric image characteristics and PM2.5 concentration through image dehazing, Region of Interest (ROI) analysis, SSIM-based feature extraction, correlation analysis, and machine learning models.

> **Note:** This repository contains selected implementation and experiments from the internship. Certain datasets, research-specific details, and experimental results have not been included in the public repository.

---

## Project Overview

The overall project workflow can be summarized as:

```text
Input Images
     │
     ▼
Image Dehazing
     │
     ▼
Image Data Processing
     │
     ▼
ROI Selection & SSIM Calculation
     │
     ▼
Extra Trees Regression
     │
     ▼
PM2.5 Prediction
     │
     └──────────────► Experimental
                       ResNet / ViT
```

---

## Repository Structure

```text
PM2.5-Prediction/
│
├── Trial_Dehaze/
│   └── Initial experiments related to image dehazing
│
├── Final_Dehaze_Code.py
│   └── Final image dehazing implementation
│
├── Image_Data_Processing.py
│   └── Image processing, CSV preparation,
│       feature preparation and SSIM calculation
│
├── Test_Correlation/
│   └── Experiments analysing the relationship
│       between SSIM-based features and PM2.5
│
├── ROI_Examples/
│   └── Examples of selected Regions of Interest
│
├── Models/
│   │
│   ├── Belur/
│   │   ├── Input CSV files
│   │   ├── Output CSV files
│   │   └── Extra Trees Regressor model
│   │
│   ├── Belur_RBU/
│   │   ├── Input CSV files
│   │   ├── Output CSV files
│   │   └── Extra Trees Regressor model
│   │
│   ├── Ghusuri/
│   │   ├── Input CSV files
│   │   ├── Output CSV files
│   │   └── Extra Trees Regressor model
│   │
│   └── ...
│
├── Experimental/
│   │
│   ├── ResNet/
│   │   └── ResNet-based experiments
│   │
│   └── ViT/
│       └── Vision Transformer experiments
│
├── requirements.txt
│
└── README.md
```

---

## 1. Image Dehazing

### `Trial_Dehaze/`

Contains the initial experiments and trial implementations carried out while investigating image dehazing techniques.

### `Final_Dehaze_Code.py`

Contains the finalized implementation used for image dehazing before subsequent feature extraction and analysis.

Image dehazing was explored to reduce the effect of atmospheric haze and improve the visual information available for further analysis.

---

## 2. Image Data Processing

### `Image_Data_Processing.py`

This script handles the preparation of image and environmental data for the prediction pipeline.

Major tasks include:

- Processing image data
- Matching images with environmental measurements
- Preparing CSV files
- Selecting Regions of Interest
- Calculating SSIM
- Preparing features for machine learning
- Organizing the processed data for model training

---

## 3. ROI Analysis

### `ROI_Examples/`

Contains examples of the **Regions of Interest (ROIs)** selected from the images.

ROI-based processing was used to focus the analysis on specific regions of an image and investigate whether these regions provided useful visual information for PM2.5 prediction.

---

## 4. SSIM and Correlation Analysis

### `Test_Correlation/`

Contains experiments investigating the relationship between image-derived features and PM2.5 concentration.

The experiments include:

- SSIM-based analysis
- Comparison of image characteristics
- PM2.5 normalization
- Correlation analysis
- Visualization of feature relationships

This stage helped evaluate the usefulness of image-derived features before using them for machine learning.

---

## 5. Machine Learning Models

### `Models/`

The primary prediction models were developed using the **Extra Trees Regressor**.

Separate directories were maintained for different datasets/locations.

For example:

```text
Models/
│
├── Belur/
├── Belur_RBU/
├── Ghusuri/
└── ...
```

Each model directory contains the corresponding input and output CSV files along with the trained model and supporting files.

### Prediction Pipeline

```text
Image-derived Features
          +
Environmental / Temporal Features
          │
          ▼
   Feature Engineering
          │
          ▼
 Extra Trees Regressor
          │
          ▼
   Model Prediction
```

---

## 6. Experimental Deep Learning Models

### `Experimental/`

Additional experiments were conducted using deep learning approaches to investigate alternative ways of extracting information from images.

### ResNet

Experiments using **ResNet-based models** were performed to explore convolutional neural network-based image feature extraction.

### Vision Transformer

Experiments using **Vision Transformer (ViT)** architectures were conducted as an alternative approach to image representation and feature extraction.

These experiments were exploratory and separate from the primary Extra Trees regression pipeline.

---

## Technologies Used

### Programming
- Python

### Machine Learning
- Scikit-learn
- Extra Trees Regressor
- Feature engineering

### Deep Learning
- PyTorch
- ResNet
- Vision Transformer (ViT)

### Image Processing
- OpenCV
- NumPy
- SSIM

### Data Processing
- Pandas
- CSV

### Visualization
- Matplotlib

---

## Key Areas of Work

- Image preprocessing
- Image dehazing
- Region of Interest (ROI) selection
- SSIM-based image analysis
- Environmental data processing
- Feature engineering
- Correlation analysis
- Extra Trees regression
- PM2.5 prediction
- ResNet-based experimentation
- Vision Transformer experimentation

---

## Outcome

The primary outcome of the internship was the development of a **machine learning-based pipeline for image-assisted PM2.5 prediction**.

The work combined image processing, feature extraction, environmental data, and machine learning to investigate the possibility of estimating PM2.5 concentration from visual information.

Additional experiments using **ResNet and Vision Transformer architectures** were also conducted to explore deep learning-based approaches for image representation.

---

## Internship

**Organization:** Indian Statistical Institute (ISI), Kolkata  
**Domain:** Computer Vision, Image Processing & Machine Learning  
**Project:** Image-Based PM2.5 Prediction

---

## Acknowledgements

I would like to thank my supervisors and the **Indian Statistical Institute, Kolkata** for providing me with the opportunity to work on this research-oriented project and gain practical experience in computer vision, image processing, and machine learning.

---

## Disclaimer

This repository is primarily intended for **academic, educational, and portfolio purposes**.

Certain datasets, research-specific implementation details, model configurations, and experimental results may not be publicly included due to project and research considerations.
