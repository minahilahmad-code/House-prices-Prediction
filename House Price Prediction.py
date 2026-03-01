# House Price Prediction - Complete Implementation
# =================================================

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. PROBLEM STATEMENT AND GOAL
# ==============================
print("="*70)
print("HOUSE PRICE PREDICTION MODEL")
print("="*70)
print("\nProblem Statement:")
print("Predict house prices based on various property features")
print("\nGoal:")
print("Build a regression model that accurately predicts house prices")
print("Evaluate model performance using MAE and RMSE metrics")
print("="*70)

# 2. DATASET LOADING
# ==================
print("\n\n2. LOADING THE DATASET")
print("="*50)

# Since we don't have the actual file, let's create a synthetic dataset
# that mimics real estate data
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'sqft_living': np.random.normal(2000, 500, n_samples).astype(int),
    'sqft_lot': np.random.normal(8000, 2000, n_samples).astype(int),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'floors': np.random.randint(1, 3, n_samples),
    'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'view': np.random.randint(0, 4, n_samples),
    'condition': np.random.randint(1, 5, n_samples),
    'grade': np.random.randint(3, 11, n_samples),
    'sqft_above': np.random.normal(1500, 400, n_samples).astype(int),
    'sqft_basement': np.random.normal(500, 300, n_samples).astype(int),
    'yr_built': np.random.randint(1900, 2020, n_samples),
    'yr_renovated': np.random.choice([0] + list(range(1950, 2020)), n_samples),
    'zipcode': np.random.choice([98101, 98102, 98103, 98104, 98105, 98106], n_samples),
    'lat': np.random.uniform(47.5, 47.7, n_samples),
    'long': np.random.uniform(-122.4, -122.2, n_samples),
}

# Create price with some realistic relationships
data['price'] = (
    data['sqft_living'] * 200 +
    data['bedrooms'] * 5000 +
    data['bathrooms'] * 10000 +
    data['waterfront'] * 200000 +
    data['view'] * 25000 +
    data['grade'] * 50000 +
    np.random.normal(0, 50000, n_samples)
)
data['price'] = np.abs(data['price'])  # Ensure positive prices

# Create DataFrame
df = pd.DataFrame(data)
print(f"Dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 3. DATA PREPROCESSING
# =====================
print("\n\n3. DATA PREPROCESSING")
print("="*50)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values (if any)
df['yr_renovated'] = df['yr_renovated'].fillna(0)

# Feature engineering
print("\nFeature Engineering:")
# Create new features
df['house_age'] = 2024 - df['yr_built']
df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
df['total_sqft'] = df['sqft_living'] + df['sqft_basement']
df['bedroom_bath_ratio'] = df['bedrooms'] / df['bathrooms']
df['price_per_sqft'] = df['price'] / df['sqft_living']

print("Created new features: house_age, has_basement, total_sqft, bedroom_bath_ratio, price_per_sqft")

# Encode categorical variables
le = LabelEncoder()
df['zipcode_encoded'] = le.fit_transform(df['zipcode'])

print(f"\nDataset shape after preprocessing: {df.shape}")

# 4. DATA VISUALIZATION AND EXPLORATION
# ======================================
print("\n\n4. DATA VISUALIZATION AND EXPLORATION")
print("="*50)

# Create figure for multiple plots
fig = plt.figure(figsize=(20, 12))

# 4.1 Distribution of target variable (price)
plt.subplot(2, 3, 1)
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')

# 4.2 Correlation heatmap
plt.subplot(2, 3, 2)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix[['price']].sort_values(by='price', ascending=False), 
            annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation with Price', fontsize=14, fontweight='bold')

# 4.3 Price vs Square Footage
plt.subplot(2, 3, 3)
sns.scatterplot(data=df, x='sqft_living', y='price', alpha=0.5)
plt.title('Price vs Living Area', fontsize=14, fontweight='bold')
plt.xlabel('Square Footage (Living)')
plt.ylabel('Price ($)')

# 4.4 Price by Number of Bedrooms
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='bedrooms', y='price')
plt.title('Price Distribution by Bedrooms', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')

# 4.5 Price by Location (Zipcode)
plt.subplot(2, 3, 5)
top_zipcodes = df.groupby('zipcode')['price'].mean().sort_values(ascending=False).head(10)
top_zipcodes.plot(kind='bar')
plt.title('Average Price by Top 10 Zipcodes', fontsize=14, fontweight='bold')
plt.xlabel('Zipcode')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)

# 4.6 Price vs Grade
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='grade', y='price')
plt.title('Price Distribution by Grade', fontsize=14, fontweight='bold')
plt.xlabel('Grade')
plt.ylabel('Price ($)')

plt.tight_layout()
plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. MODEL TRAINING AND EVALUATION
# ================================
print("\n\n5. MODEL TRAINING AND EVALUATION")
print("="*50)

# Select features for modeling
feature_columns = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors', 
                   'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                   'sqft_basement', 'house_age', 'has_basement', 'total_sqft', 
                   'bedroom_bath_ratio', 'zipcode_encoded', 'lat', 'long']

X = df[feature_columns]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5.1 Linear Regression Model
print("\n5.1 Linear Regression Model:")
print("-" * 30)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

# Evaluation metrics for Linear Regression
lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))

print(f"Training MAE: ${lr_train_mae:,.2f}")
print(f"Test MAE: ${lr_test_mae:,.2f}")
print(f"Training RMSE: ${lr_train_rmse:,.2f}")
print(f"Test RMSE: ${lr_test_rmse:,.2f}")

# Cross-validation
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE (mean ± std): ${-lr_cv_scores.mean():,.2f} ± ${lr_cv_scores.std():,.2f}")

# 5.2 Gradient Boosting Model
print("\n5.2 Gradient Boosting Model:")
print("-" * 30)

gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Predictions
gb_train_pred = gb_model.predict(X_train_scaled)
gb_test_pred = gb_model.predict(X_test_scaled)

# Evaluation metrics for Gradient Boosting
gb_train_mae = mean_absolute_error(y_train, gb_train_pred)
gb_test_mae = mean_absolute_error(y_test, gb_test_pred)
gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))

print(f"Training MAE: ${gb_train_mae:,.2f}")
print(f"Test MAE: ${gb_test_mae:,.2f}")
print(f"Training RMSE: ${gb_train_rmse:,.2f}")
print(f"Test RMSE: ${gb_test_rmse:,.2f}")

# Cross-validation
gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE (mean ± std): ${-gb_cv_scores.mean():,.2f} ± ${gb_cv_scores.std():,.2f}")

# Feature importance for Gradient Boosting
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 6. VISUALIZE PREDICTIONS VS ACTUAL
# ===================================
print("\n\n6. VISUALIZING PREDICTIONS VS ACTUAL")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Linear Regression - Training
axes[0, 0].scatter(y_train, lr_train_pred, alpha=0.5)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Prices ($)')
axes[0, 0].set_ylabel('Predicted Prices ($)')
axes[0, 0].set_title(f'Linear Regression - Training Set\nMAE: ${lr_train_mae:,.2f}')

# Linear Regression - Test
axes[0, 1].scatter(y_test, lr_test_pred, alpha=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Prices ($)')
axes[0, 1].set_ylabel('Predicted Prices ($)')
axes[0, 1].set_title(f'Linear Regression - Test Set\nMAE: ${lr_test_mae:,.2f}')

# Gradient Boosting - Training
axes[1, 0].scatter(y_train, gb_train_pred, alpha=0.5)
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Prices ($)')
axes[1, 0].set_ylabel('Predicted Prices ($)')
axes[1, 0].set_title(f'Gradient Boosting - Training Set\nMAE: ${gb_train_mae:,.2f}')

# Gradient Boosting - Test
axes[1, 1].scatter(y_test, gb_test_pred, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Prices ($)')
axes[1, 1].set_ylabel('Predicted Prices ($)')
axes[1, 1].set_title(f'Gradient Boosting - Test Set\nMAE: ${gb_test_mae:,.2f}')

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. MODEL COMPARISON
# ===================
print("\n\n7. MODEL COMPARISON")
print("="*50)

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Gradient Boosting'],
    'Train MAE': [f"${lr_train_mae:,.2f}", f"${gb_train_mae:,.2f}"],
    'Test MAE': [f"${lr_test_mae:,.2f}", f"${gb_test_mae:,.2f}"],
    'Train RMSE': [f"${lr_train_rmse:,.2f}", f"${gb_train_rmse:,.2f}"],
    'Test RMSE': [f"${lr_test_rmse:,.2f}", f"${gb_test_rmse:,.2f}"],
    'CV MAE': [f"${-lr_cv_scores.mean():,.2f}", f"${-gb_cv_scores.mean():,.2f}"]
})

print(comparison_df.to_string(index=False))

# Determine best model
if gb_test_mae < lr_test_mae:
    best_model = "Gradient Boosting"
    best_mae = gb_test_mae
else:
    best_model = "Linear Regression"
    best_mae = lr_test_mae

print(f"\n✅ Best Model: {best_model} with Test MAE of ${best_mae:,.2f}")

# 8. SAVE THE BEST MODEL
# ======================
print("\n\n8. SAVING THE BEST MODEL")
print("="*50)

import joblib

# Save the best model and scaler
if best_model == "Gradient Boosting":
    joblib.dump(gb_model, 'best_house_price_model.pkl')
else:
    joblib.dump(lr_model, 'best_house_price_model.pkl')

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print(f"✅ Best model saved as 'best_house_price_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("✅ Feature columns saved as 'feature_columns.pkl'")

# 9. EXAMPLE PREDICTION
# =====================
print("\n\n9. EXAMPLE PREDICTION")
print("="*50)

# Create a sample house
sample_house = {
    'sqft_living': 2500,
    'sqft_lot': 10000,
    'bedrooms': 4,
    'bathrooms': 2.5,
    'floors': 2,
    'waterfront': 0,
    'view': 1,
    'condition': 3,
    'grade': 7,
    'sqft_above': 1800,
    'sqft_basement': 700,
    'house_age': 15,
    'has_basement': 1,
    'total_sqft': 3200,
    'bedroom_bath_ratio': 1.6,
    'zipcode_encoded': 2,
    'lat': 47.6,
    'long': -122.3
}

# Convert to DataFrame and scale
sample_df = pd.DataFrame([sample_house])
sample_scaled = scaler.transform(sample_df[feature_columns])

# Make prediction
if best_model == "Gradient Boosting":
    predicted_price = gb_model.predict(sample_scaled)[0]
else:
    predicted_price = lr_model.predict(sample_scaled)[0]

print(f"Sample House Features:")
for key, value in sample_house.items():
    print(f"  {key}: {value}")
print(f"\n💰 Predicted Price: ${predicted_price:,.2f}")

print("\n" + "="*70)
print("✅ HOUSE PRICE PREDICTION TASK COMPLETED SUCCESSFULLY!")
print("="*70)