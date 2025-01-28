# Combining freely available data and IoT devices to
predict drought in Eastern and Central Europe

## Abstract
This study introduces a novel approach to drought prediction in Eastern and Central Europe by combining IoT soil moisture sensor data with freely available datasets. It integrates real-time IoT soil moisture measurements, SOTER-based soil parameters (SOTWIS), and NASA POWER environmental data. The methodology includes data preprocessing, spatial clustering using DBSCAN, and a machine learning framework combining Random Forest, XGBoost, and LSTM networks. The research highlights how incorporating soil texture data with climatic variables enhances drought prediction accuracy. It also presents an Arduino-based IoT prototype for continuous soil monitoring, addressing the benefits and challenges of IoT in agriculture. This approach offers improved drought prediction and supports better agricultural decision-making in under-researched regions.

## Overview

The system combines three key data sources:
- SOTER-based soil parameter estimates (SOTWIS) containing detailed soil characteristics
- NASA POWER data providing environmental information and soil humidity estimates
- IoT soil moisture sensor data for real-time ground measurements

The prediction framework utilizes a combination of Random Forest, XGBoost, and LSTM networks to generate accurate drought predictions while considering both soil properties and environmental factors.

## Features

- Data fusion of multiple environmental and soil datasets
- Spatial clustering using DBSCAN for handling missing data
- Multi-model machine learning framework combining:
  - Random Forest for robust predictions
  - XGBoost for capturing nonlinear relationships
  - LSTM networks for temporal pattern recognition
- Arduino-based IoT integration for real-time soil moisture monitoring
- Comprehensive data preprocessing pipeline
- Spatial and temporal data alignment capabilities

## Prerequisites

- Python 3.8+
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - tensorflow
  - geopandas
  - matplotlib
  - Arduino IDE (for IoT component)


# Model Development
The implemented system integrates traditional machine learning models (Random Forest and XGBoost) and a deep learning architecture (Long Short-Term Memory, LSTM) to leverage their respective strengths in handling temporal and nonlinear patterns in data.

The model pipeline begins with preprocessing the input dataset, which contains spatially and temporally distributed features such as latitude (LAT), longitude (LON), year, and monthly variables (e.g., soil moisture and precipitation). The approach groups data by spatial coordinates and sorts it chronologically to ensure temporal consistency.

To prepare data for temporal modeling, a sliding-window approach is adopted, generating sequential input-output pairs. For each location, a sequence of data spanning a specified lookback period (e.g., 3 months) serves as the input, while the subsequent month's values form the target. This method captures the temporal dependencies necessary for drought prediction. The input features are scaled using the StandardScaler, ensuring uniform treatment of variables and accelerating convergence during training.

Random Forest (RF) is used as one of the benchmark models. Its ensemble nature, combining multiple decision trees, enables robust predictions even in the presence of noisy or limited data. The RF model is trained on flattened versions of the temporal sequences, ensuring compatibility with its non-sequential learning architecture.

Extreme Gradient Boosting (XGBoost) is another tree-based ensemble model utilized in this study. Known for its ability to capture complex nonlinear relationships and interaction effects, XGBoost optimizes model performance through regularization and efficient gradient computation. It processes flattened temporal sequences similar to Random Forest.

The LSTM architecture is designed to explicitly handle sequential data. The network comprises two LSTM layers, each followed by a Dropout layer to mitigate overfitting. The first LSTM layer outputs sequences to preserve temporal information for subsequent processing, while the second aggregates these sequences into a single representation. Finally, a dense layer with 12 output neurons predicts monthly drought-related indicators for an entire year. The LSTM network is trained using the Adam optimizer, with mean squared error (MSE) as the loss function.

Copyright © by the paper’s authors. Copying permitted for
private and academic purposes.
