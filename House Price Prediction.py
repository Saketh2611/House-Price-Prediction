import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file = "HousePricePrediction (1).xlsx" #replace with correct path if needed 
dataset = pd.read_excel(file)

# Display the first 5 rows
print(dataset.head())

# Dataset shape
print("Dataset shape:", dataset.shape)

# Identifying categorical and numerical variables
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

print("Categorical variables:", len(object_cols))
print(object_cols)

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Correlation heatmap
numerical_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(), cmap='magma', fmt='.2f', linewidths=2, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Unique values of categorical features
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].nunique())

plt.figure(figsize=(10, 6))
plt.title('No. of Unique Values of Categorical Features')
plt.xticks(rotation=45)
sns.barplot(x=object_cols, y=unique_values)
plt.xlabel("Categorical Features")
plt.ylabel("Number of Unique Values")
plt.show()

# Distribution of categorical features
plt.figure(figsize=(18, 36))
plt.suptitle('Categorical Features: Distribution', fontsize=16)
for index, col in enumerate(object_cols, 1):
    plt.subplot(11, 4, index)
    sns.barplot(x=dataset[col].value_counts().index, y=dataset[col].value_counts().values)
    plt.xticks(rotation=90)
    plt.title(col)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# Data cleaning
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()

# Re-identify categorical variables
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:", object_cols)
print('Number of categorical features:', len(object_cols))

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Train-test split
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Support Vector Regressor
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_preds = model_SVR.predict(X_valid)
print("SVR MAE:", mean_absolute_error(Y_valid, Y_preds))

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred_rf = model_RFR.predict(X_valid)
print("Random Forest MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_rf))

# Linear Regression
from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_lr = model_LR.predict(X_valid)
print("Linear Regression MAPE:", mean_absolute_percentage_error(Y_valid, Y_pred_lr))

# Actual vs Predicted plot for SVR
plt.figure(figsize=(8, 6))
plt.scatter(Y_valid, Y_preds)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (SVR)")
plt.show()

# Predicting for a new house
new_house = pd.DataFrame({
    'MSSubClass': [110],
    'MSZoning': ["RM"],
    'LotArea': [9000],
    'LotConfig': ["FR2"],
    'BldgType': ["1Fam"],
    'OverallCond': [5],
    'YearBuilt': [2005],
    'YearRemodAdd': [2005],
    'Exterior1st': ["Wd Sdng"],
    'BsmtFinSF2': [30],
    'TotalBsmtSF': [1000]
})

# One-hot encoding for the new house
new_house_encoded = new_house.copy()
OH_cols_new = pd.DataFrame(OH_encoder.transform(new_house[object_cols]))
OH_cols_new.columns = OH_encoder.get_feature_names_out()
OH_cols_new.index = new_house.index

new_house_encoded.drop(object_cols, axis=1, inplace=True)
new_house_encoded = pd.concat([new_house_encoded, OH_cols_new], axis=1)

# Align columns with training data
new_house_encoded = new_house_encoded.reindex(columns=X_train.columns, fill_value=0)

# Prediction
predicted_price = model_SVR.predict(new_house_encoded)
print("Predicted price for new house:", predicted_price[0])
