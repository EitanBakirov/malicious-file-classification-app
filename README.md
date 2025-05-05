# Malware Detection: ML Classification & Interactive Application

## Overview

This project consists of two main components:
1. **Data Science Project**: A comprehensive malicious file classification pipeline
2. **Interactive Web Application**: A Streamlit app for real-time file analysis and classification

The goal is to classify files as either malicious or benign based on static analysis features. The project combines rigorous data science methodology with a user-friendly interface for practical application.

For an in-depth explanation of our process, see ["report_english.pdf"/"report_hebrew.pdf"](https://github.com/EitanBakirov/malicious-file-classification/blob/main/report_english.pdf) or explore the complete [notebook.ipynb](https://github.com/EitanBakirov/malicious-file-classification/blob/main/notebook.ipynb).

## Table of Contents

- [Malware Detection: ML Classification \& Interactive Application](#malware-detection-ml-classification--interactive-application)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Components](#project-components)
    - [1. Data Science Pipeline](#1-data-science-pipeline)
    - [2. Interactive Application](#2-interactive-application)
  - [Project Description](#project-description)
  - [Data](#data)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Installation](#installation)
  - [Authors](#authors)
  - [Contact](#contact)

## Project Components

### 1. Data Science Pipeline

The complete machine learning pipeline includes:

- **Part 1 - Exploring the Data**: Comprehensive EDA of file characteristics
- **Part 2 - Preprocessing**: 
  - Handling Missing Values
  - Handling Categorical Features
  - Handling Outliers
  - Large Dimensionality
  - Data Normalizing
  - Dimensionality Reduction
- **Part 3 - Running the Models**:
  - Two Simple Models (KNN, Logistic Regression)
  - Two Advanced Models (ANN, Random Forest)
- **Part 4 - Evaluation**:
  - Confusion Matrix
  - Advanced Metrics
  - KFold Cross-Validation
- **Part 5 - Prediction**: Final model deployment

### 2. Interactive Application

Our Streamlit application provides:
- User-friendly interface for file upload and analysis
- Real-time classification of files as malicious or benign
- Model information and performance metrics
- Adjustable detection threshold for customized sensitivity

## Project Description

In this project, we were tasked with classifying files as malicious (1) or non-malicious (0) based on various features in the dataset. The project involved:

- Exploratory Data Analysis to understand the data's distribution and correlations.
- Data preprocessing, including handling missing values, dealing with categorical features, and feature engineering.
- Building and evaluating machine learning models, including Random Forest and K-Nearest Neighbors, to select the best model.
- The selected Random Forest model achieved a high AUC score and was used for predictions.

Also, in the folder "Instructions" you can get the full instructions of the project in hebrew and english.

## Data

The dataset contains 60,000 observations classified as malicious or non-malicious files. Some features are known, while others are anonymous. Data preprocessing steps included handling missing values, normalizing data, and dealing with outliers.

## Methodology

We applied exploratory data analysis, including histograms and correlation analysis, to understand feature distributions and relationships. Data preprocessing involved filling missing values, converting categorical features, and feature engineering. We built machine learning models, including Random Forest and K-Nearest Neighbors, using cross-validation and hyperparameter tuning.

## Results

The Random Forest model was selected as the best-performing model with a high AUC score. The model showed a balanced trade-off between precision and recall. Feature importance analysis revealed key features contributing to classification, such as 'Avlength,' 'B,' 'imports,' 'Urls,' and 'file_type_prob_trid.'

## Installation

To reproduce the project and run the application:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EitanBakirov/malicious-file-classification.git
   cd malicious-file-classification
   ```

2. **Create a virtual environment**:
    ```bash
    # Create a new virtual environment
    python -m venv .venv

    # Activate the virtual environment
    # On Windows:
    .venv\Scripts\activate

    # On macOS/Linux:
    source .venv/bin/activate
    ```

3. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```


4. **Run the Streamlit application:**:
  ```bash
  streamlit run app.py
  ```

  The application will open in your default web browser.
  
## Authors

- [Eitan Bakirov](https://github.com/EitanBakirov)
- [Yuval Bakirov](https://github.com/YuvalBakirov)

## Contact

For questions or feedback related to the project, please contact:<br>
Eitan Bakirov at EitanBakirov@gmail.com <br>
Yuval Bakirov at yuvalbakirov@gmail.com
