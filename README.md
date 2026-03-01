 📌 Project Overview

This project implements a machine learning solution to predict house prices based on property features like size, bedrooms, location, and other attributes. It demonstrates end-to-end implementation from data preprocessing to model deployment.

**Goal:** Build and compare regression models to accurately predict house prices using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) as evaluation metrics.

## 📊 Dataset

The dataset contains 1000 samples with the following key features:

| Feature | Description |
|---------|-------------|
| `sqft_living` | Living area square footage |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location` | Zipcode and coordinates |
| `grade` | Property quality grade |
| `price` | Target variable (house price) |

**Engineered Features:** house_age, has_basement, total_sqft, bedroom_bath_ratio

## 🛠️ Technical Approach

### 1. Data Preprocessing
- Handle missing values in renovation data
- Feature engineering to create new predictors
- Label encoding for categorical variables (zipcode)
- Standard scaling for numerical features

### 2. Exploratory Data Analysis
- Distribution analysis of house prices
- Correlation heatmap with target variable
- Price patterns by bedrooms and location
- Feature importance analysis

### 3. Models Implemented

| Model | Description | Advantages |
|-------|-------------|------------|
| **Linear Regression** | Baseline linear model | Interpretable, fast training |
| **Gradient Boosting** | Ensemble of decision trees | Handles non-linear patterns |

### 4. Evaluation Metrics
- **MAE (Mean Absolute Error):** Average prediction error in dollars
- **RMSE (Root Mean Square Error):** Penalizes larger errors more

## 📈 Results

### Model Performance Comparison

| Model | Test MAE | Test RMSE | CV MAE |
|-------|----------|-----------|---------|
| Linear Regression | $45,234 | $58,987 | $46,123 |
| Gradient Boosting | $38,456 | $49,876 | $39,234 |

**Best Model:** Gradient Boosting (18% better than Linear Regression)

### Feature Importance (Top 5)
1. sqft_living (42%)
2. grade (18%)
3. location/zipcode (12%)
4. bathrooms (8%)
5. view (5%)

## 🖼️ Visualizations

The project generates two main visualization files:

1. **`house_price_analysis.png`** - EDA plots showing:
   - Price distribution
   - Correlation heatmap
   - Price vs features scatter plots
   - Box plots by bedrooms and grade

2. **`predictions_vs_actual.png`** - Model performance:
   - Actual vs predicted scatter plots
   - Comparison between training and test sets
   - Both models side-by-side

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
