# PM2.5 Estimation from Distant Views Using Machine Learning

### Across the Miles with Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv" />
  <img src="https://img.shields.io/badge/Machine%20Learning-ExtraTrees-orange" />
  <img src="https://img.shields.io/badge/Research-Computer%20Vision-purple" />
</p>

## Overview

Air pollution, particularly fine particulate matter (PM2.5), is a major environmental and public health concern. Conventional air quality monitoring stations are expensive to install and maintain, making widespread monitoring challenging.

This project investigates an image-based machine learning approach for estimating PM2.5 concentrations at different distances from a single camera.

The proposed methodology uses atmospheric visibility degradation as a visual indicator of particulate matter concentration. It combines image dehazing, Region of Interest (ROI) extraction, Structural Similarity Index Measure (SSIM), temporal feature engineering and machine learning regression.

Unlike conventional image-based approaches that estimate a single PM2.5 value from an entire image, this research explores the estimation of pollution levels at multiple distances using different camera views.

## Research Objectives

- Estimate PM2.5 concentrations using images captured from distant locations.
- Investigate the relationship between atmospheric visibility degradation and PM2.5 concentration.
- Extract meaningful visual features using predefined Regions of Interest (ROI).
- Investigate the contribution of SSIM towards PM2.5 prediction.
- Develop regression models incorporating nearby and distant visual information.
- Explore residual prediction to improve initial PM2.5 estimates.
- Evaluate model performance against ground-truth monitoring station measurements.

## Study Locations and Data Acquisition

Images were acquired using a Dahua 4 MP 32X Network PTZ Camera installed on the terrace of the Indian Statistical Institute (ISI), Kolkata.

Three monitoring locations were considered:

| Monitoring Station | Approximate Distance from ISI |
|---|---:|
| Rabindra Bharati University (RBU) | 2.6 km |
| Belur | 4 km |
| Ghusuri (Howrah) | 8 km |

PM2.5 measurements obtained from the corresponding monitoring stations were used as ground-truth data.

## Project Workflow

The implementation follows the following processing pipeline:

```text
             Original Images
                    |
                    v
             Image Organization
                    |
                    v
              ROI Extraction
                    |
                    v
              Image Dehazing
                    |
                    v
          Grayscale Conversion
                    |
                    v
         Morphological Erosion
                    |
                    v
            SSIM Calculation
                    |
                    v
          Feature Extraction
                    |
                    v
        Dataset Preparation
                    |
                    v
        Temporal Feature Encoding
                    |
                    v
          ExtraTrees Regression
                    |
                    v
           Residual Prediction
                    |
                    v
          Final PM2.5 Estimation
```

## Repository Contents

The repository contains four primary implementations.

### 1. Image Dehazing

**File:** `dehaze_production.py`

This script implements a pyramid-fusion-based image dehazing technique to enhance visibility and recover image details.

The implementation generates two image representations:

- White-balanced image.
- Contrast-enhanced image.

These representations are processed using multiple weight maps before multi-scale image fusion.

**Image enhancement techniques**

- White balancing.
- Contrast enhancement.
- Gaussian smoothing.
- High-boost sharpening.
- Wavelet-based fusion.
- Contrast Limited Adaptive Histogram Equalization (CLAHE).

**Weight map generation**

Six weight maps are calculated for each image representation:

| Weight Map | Purpose |
|---|---|
| Visibility | Measures local visibility-related image variations |
| Visual Saliency | Extracts frequency and colour-based saliency information |
| Entropy | Measures local information content |
| Chromaticity | Captures colour-related information |
| Colorfulness | Measures local colourfulness |
| Luminance | Represents luminance variations |

**Multi-scale image fusion**

The implementation uses:

- Gaussian pyramid generation.
- Laplacian pyramid generation.
- Pyramid reduction and expansion.
- Weighted image fusion.
- Pyramid reconstruction.

The final enhanced image is generated through multi-scale fusion of the two processed image representations.

The main dehazing function is:

`pyramidFusionDehaze(img, ps)`

The script also includes batch processing to:

1. Read images from location-specific folders.
2. Apply the dehazing algorithm.
3. Save processed images with the `_d.png` suffix.
4. Preserve the original location-wise folder organization.

### 2. Image Data Processing and Dataset Preparation

**File:** `image_data_process.py`

This script handles image organization, monitoring-station data cleaning and the generation of image-associated environmental datasets.

**Image organization**

- Renaming images using acquisition date and time.
- Checking the correspondence between original and dehazed images.
- Organizing image filenames for subsequent processing.

**Air quality data preprocessing**

Monitoring-station Excel files are processed to extract:

- Measurement start date.
- Measurement end date.
- Start time.
- End time.
- Relative Humidity (RH).
- Wind Speed (WS).
- Air Temperature (AT).
- PM2.5 concentration.

Data cleaning includes:

- Date and time conversion.
- Numerical conversion of environmental measurements.
- Removal of records containing missing PM2.5, RH, WS or AT values.

**Image and monitoring data alignment**

The `create_dataset()` function associates image filenames with corresponding monitoring-station records.

The image acquisition date and time are extracted from filenames and matched against the monitoring station's measurement intervals.

The resulting dataset contains:

| Column | Description |
|---|---|
| ImageFile | Name of the corresponding image |
| RH | Relative humidity |
| WS | Wind speed |
| AT | Air temperature |
| PM2.5 | Ground-truth PM2.5 concentration |

The processed datasets are exported in CSV format for subsequent feature extraction and machine learning experiments.

### 3. ROI Extraction and SSIM Calculation

**File:** `roi_trial.py`

This script extracts predefined image regions and calculates structural similarity features using original and dehazed images.

**Processing steps**

1. Read the prepared image-associated CSV dataset.
2. Load the original image.
3. Load the corresponding dehazed image.
4. Convert both images into grayscale.
5. Apply morphological erosion to the dehazed grayscale image.
6. Extract the predefined ROI using pixel coordinates.
7. Calculate SSIM between the original and dehazed ROI.
8. Calculate SSIM between the original and eroded dehazed ROI.
9. Append the extracted features to the CSV dataset.

**ROI Configuration**

The implemented ROI uses the following image coordinates:

| Parameter | Value |
|---|---:|
| Starting row | 378 |
| Ending row | 1050 |
| Starting column | 1570 |
| Ending column | 2215 |

**Morphological Operation**

A rectangular structuring element of size 3 × 3 is used for morphological erosion.

**Extracted Features**

| Feature | Description |
|---|---|
| `SSIM_ROI` | Structural similarity between the original ROI and dehazed ROI |
| `ERODE_SSIM_ROI` | Structural similarity between the original ROI and eroded dehazed ROI |

The extracted features are stored in the CSV dataset and subsequently used as inputs for machine learning models.

### 4. Machine Learning and Residual Prediction

**File:** `model.ipynb`

This notebook implements ExtraTrees regression models for PM2.5 estimation using image-derived features and temporal information.

The uploaded notebook includes Belur and Belur–FRB experiments, followed by a residual prediction model for Belur.

**A. Temporal Feature Engineering**

Date and time information are extracted from image filenames:

- Day
- Month
- Hour
- Minute

Month names are converted into numerical values.

Cyclic encoding is applied to represent periodic temporal information using sine and cosine transformations.

The notebook calculates:

- `Day_sin`, `Day_cos`
- `Mon_sin`, `Mon_cos`
- `Hour_sin`, `Hour_cos`
- `Min_sin`, `Min_cos`

The regression feature matrices use SSIM features along with cyclically encoded day, month and hour information.

**B. Feature Selection**

The primary model inputs are:

- `SSIM_ROI`
- `ERODE_SSIM_ROI`
- `Day_sin`
- `Day_cos`
- `Mon_sin`
- `Mon_cos`
- `Hour_sin`
- `Hour_cos`

The target variable is normalized PM2.5 concentration.

**C. Model Training**

ExtraTrees Regressor is used for PM2.5 regression.

The dataset is divided into training and testing subsets using an 80:20 split with `random_state=42`.

Hyperparameter optimization is performed using GridSearchCV.

The parameter grid includes:

| Parameter | Values |
|---|---|
| `n_estimators` | 10 to 1010, in steps of 10 |
| `max_features` | sqrt, log2, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 |
| `bootstrap` | True, False |

The notebook trains separate models for the Belur and Belur–FRB datasets.

**D. Residual Prediction**

A two-phase prediction strategy is implemented.

**Phase I – Initial Estimation**

- Train an ExtraTrees model using distant-view image features.
- Estimate the PM2.5 concentration corresponding to the nearby monitoring station.
- Generate initial PM2.5 estimates.

**Phase II – Residual Estimation**

- Merge the corresponding Belur and Belur–FRB datasets using image filenames.
- Calculate the residual between the actual Belur PM2.5 concentration and the initial Belur estimate.
- Normalize the residual values.
- Train a second ExtraTrees regression model to predict the residual.
- Use the image-derived features, temporal features and normalized nearby PM2.5 estimate as inputs.

The final PM2.5 estimate is calculated as:

`Final PM2.5 = Initial PM2.5 Estimate + Predicted Residual`

**E. Model Saving**

Trained regression models are saved using Joblib in `.pkl` format.

The notebook includes:

- `etr_belur.pkl`
- `etr_belur_frb.pkl`
- `etr_belur_res.pkl`

**F. Model Evaluation**

Model performance is evaluated using:

- R² score.
- Training and testing scores.
- Actual versus predicted PM2.5 comparisons.
- Scatter plots with a reference diagonal.

The notebook also exports processed training and testing datasets containing the prediction results.

## Experimental Results

The research paper reports approximately 2,500 samples collected during March–April 2026 for the distant-view experiments.

An 80:20 training-testing split was used.

### Performance Comparison

The following results are reported in the research paper:

| Experiment | Approach | Location | R² | RMSE |
|---|---|---|---:|---:|
| 1 | Algorithm I | Belur | 0.90 | 2.78 |
| 2 | Algorithm I | Howrah | 0.88 | 4.09 |
| 3 | Algorithm II | Belur | 0.92 | 2.43 |
| 4 | Algorithm II | Howrah | 0.98 | 1.57 |

*These are paper-reported experimental results, not a claim that every experiment is implemented in the uploaded notebook.*

### Key Findings

- The residual prediction approach performs better than the far-near integrated approach in the reported experiments.
- Combining visual features and temporal information improves prediction performance.
- The inclusion of nearby and distant visual information helps account for spatial variation in PM2.5 concentration.

## Technologies and Libraries

| Category | Libraries / Tools |
|---|---|
| Programming Language | Python |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Image Analysis | scikit-image |
| Scientific Computing | SciPy |
| Wavelet Processing | PyWavelets |
| Machine Learning | scikit-learn |
| Model Persistence | Joblib |
| Visualization | Matplotlib |
| Notebook Environment | Jupyter Notebook |
| Image Handling | Pillow |

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pm25-estimation-from-distant-views.git

cd pm25-estimation-from-distant-views
```

Install the required Python packages:

```bash
pip install numpy pandas opencv-python matplotlib scipy scikit-image PyWavelets scikit-learn joblib Pillow openpyxl
```

## Running the Project

The processing stages should be executed in the following order:

**Step 1: Image Dehazing**

Run `dehaze_production.py` after configuring the original image directory and output directory.

**Step 2: Dataset Preparation**

Run `image_data_process.py` after providing the monitoring-station Excel files and corresponding image directories.

**Step 3: ROI and SSIM Extraction**

Run `roi_trial.py` after preparing the original and dehazed image directories and the corresponding CSV dataset.

**Step 4: Machine Learning**

Open `model.ipynb` in Jupyter Notebook and provide the required CSV files.

Execute the notebook cells to perform feature engineering, model training, residual prediction and evaluation.

**Note:** The uploaded scripts currently contain machine-specific file paths. These paths must be updated to match the local dataset directories. The ROI script also contains an unfinished second-ROI section that needs to be completed or disabled before running the script.

## Dataset Availability

The original image dataset consists of a large collection of images captured from different monitoring locations.

Due to storage limitations, the original image collection is not included in this repository.

The repository focuses on sharing the image-processing implementation, dataset preparation code, feature extraction methodology and machine learning experiments.

Access to the corresponding image data and ground-truth monitoring station measurements is required to reproduce the complete experiments.

## Future Scope

- Extend the dataset using long-term image acquisition.
- Investigate Vision Transformers and other deep learning architectures.
- Explore smartphone-based image acquisition.
- Extend the methodology to more than two locations at different distances.
- Investigate calibration and adaptation for practical deployment.

## Research Publication

**Title:** Determination of PM2.5 from Distant Views: Across the Miles with Machine Learning

**Authors:**
- Sarbani Palit
- Sagarnil Chakraborty
- Joydip Rakshit
- Priyasha Haldar
- Anushka Mukherjee

**Research Affiliations:**
- Indian Statistical Institute, Kolkata
- Academy of Technology
- Visva-Bharati, Santiniketan
- Sikkim Manipal Institute of Technology

## Acknowledgements

The authors acknowledge the infrastructural support provided by the Indian Statistical Institute and partial funding received from the West Bengal Pollution Control Board.

## References

1. Chakraborty, S., Palit, S. and Bhandari, H. (2024). *Can an Image Tell the Tale: Looking Beyond the Haze to Determine PM2.5 Concentration.* IEEE International Joint Conference on Neural Networks.

2. Bhandari, H. and Palit, S. (2021). *Looking Beyond the Haze: A Pyramid Fusion Approach.* ICONIP 2021.

3. Gu, K., Qiao, J. and Li, X. (2019). *Highly Efficient Picture-Based Prediction of PM2.5 Concentration.* IEEE Transactions on Industrial Electronics.

4. Zhang, K. et al. (2016). *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.*

---

**Research Project:** Image-Based PM2.5 Estimation Using Machine Learning

**Institution:** Visva-Bharati, Santiniketan

**Research Collaboration:** Indian Statistical Institute, Kolkatatation details, model configurations, and experimental results may not be publicly included due to project and research considerations.
