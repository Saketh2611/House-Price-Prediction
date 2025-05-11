# 🏠 House Price Prediction using Machine Learning

This project applies machine learning models to predict house prices using features from a real estate dataset. It involves data preprocessing, exploratory data analysis (EDA), encoding categorical variables, model training, and evaluation using Support Vector Regression (SVR), Random Forest Regressor, and Linear Regression.

---

## 📁 Dataset

The dataset used in this project is an Excel file named `HousePricePrediction (1).xlsx`, which contains various features of houses and their sale prices. 

### Features include:
- Numerical features: `LotArea`, `YearBuilt`, `TotalBsmtSF`, etc.
- Categorical features: `MSZoning`, `BldgType`, `Exterior1st`, etc.
- Target variable: `SalePrice`

---

## 🧰 Libraries Used

- `pandas` – for data manipulation
- `matplotlib`, `seaborn` – for visualization
- `scikit-learn` – for preprocessing, model training, and evaluation

---

## 📊 Exploratory Data Analysis

- Identified and visualized numerical correlations using a heatmap.
- Counted unique values for each categorical feature.
- Plotted distributions of categorical feature values.

---

## 🧹 Data Preprocessing

- Removed the `Id` column.
- Imputed missing values in `SalePrice` with the mean.
- Dropped remaining rows with missing values.
- One-hot encoded categorical features using `OneHotEncoder`.

---

## 🔍 Model Training and Evaluation

Split the dataset into training (80%) and validation (20%) sets, then trained the following models:

| Model                | Metric                         | Value         |
|---------------------|--------------------------------|---------------|
| **SVR**             | Mean Absolute Error (MAE)      | Printed in output |
| **Random Forest**   | Mean Absolute Percentage Error | Printed in output |
| **Linear Regression**| Mean Absolute Percentage Error| Printed in output |

Also plotted **Actual vs Predicted Price** for SVR.

---

## 🏡 New House Price Prediction

A custom house data sample is prepared and encoded to match the model input, then predicted using the trained SVR model.

```python
predicted_price = model_SVR.predict(new_house_encoded)
