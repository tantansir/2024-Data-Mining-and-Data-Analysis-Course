# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:34:16 2024

@author: 17487
"""
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
# 定义数据文件夹路径
data_folder = 'T1DM'

# 初始化一个空的数据框
data = pd.DataFrame()

# 遍历文件夹中的每个Excel文件
for file in os.listdir(data_folder):
    if file.endswith('.xlsx'):
        file_path = os.path.join(data_folder, file)
        df = pd.read_excel(file_path)
        data = pd.concat([data, df], ignore_index=True)

# 查看合并后的数据
print(data.head())
# 转换日期时间特征
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d %H:%M:%S')
data['hour'] = data['Date'].dt.hour

# 处理缺失值
data.fillna(method='ffill', inplace=True)
# 特征选择
features = ['Insulin dose - s.c.', 'Non-insulin hypoglycemic agents', 'CSII - bolus insulin (Novolin R, IU)',
            'Insulin dose - i.v.', 'take_food', 'Age (years)', 'Height (m)', 'Weight (kg)', 'BMI (kg/m2)',
            'Smoking History (pack year)', 'Alcohol Drinking History (drinker/non-drinker)', 'Type of Diabetes',
            'Duration of Diabetes (years)', 'Acute Diabetic Complications', 'Hypoglycemia (yes/no)', 'Gender (Female=0, Male=1)',
            'hour', 'CGM_lag1', 'CGM_lag2']
# 定义创建滞后特征的函数
def create_lagged_features(df, target_col, lags):
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(-lag)
    return df

# 创建目标标签
data = create_lagged_features(data, 'CGM (mg / dl)', lags=[15, 30, 45, 60])

# 去除包含NaN值的行（因为我们创建了滞后特征）
data.dropna(inplace=True)
# 初始化训练集和测试集
X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()
# 遍历每个文件，进行数据分割
for file in os.listdir(data_folder):
    if file.endswith('.xlsx'):
        file_path = os.path.join(data_folder, file)
        df = pd.read_excel(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M:%S')
        df['hour'] = df['Date'].dt.hour
        df = create_lagged_features(df, 'CGM (mg / dl)', lags=[15, 30, 45, 60])
        df.dropna(inplace=True)
        
        X_file = df[features]
        y_file = df[['CGM (mg / dl)_lag15', 'CGM (mg / dl)_lag30', 'CGM (mg / dl)_lag45', 'CGM (mg / dl)_lag60']]
        
        # 分割数据
        X_train = pd.concat([X_train, X_file.iloc[:-4, :]], ignore_index=True)
        y_train = pd.concat([y_train, y_file.iloc[:-4, :]], ignore_index=True)
        
        X_test = pd.concat([X_test, X_file.iloc[-4:, :]], ignore_index=True)
        y_test = pd.concat([y_test, y_file.iloc[-4:, :]], ignore_index=True)
# 处理分类特征（如果有）
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# 确保训练集和测试集具有相同的列
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
# 创建预处理和模型管道
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), X_train.columns)
])

# 创建SVM回归模型
svm = SVR()

# 创建多输出回归器
multi_output_svr = MultiOutputRegressor(svm)

# 将预处理器和模型组成管道
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', multi_output_svr)
])
# 定义参数网格
param_grid = {
    'regressor__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'regressor__estimator__C': [0.1, 1, 10, 100],
    'regressor__estimator__epsilon': [0.1, 0.2, 0.5, 1]
}

# 使用GridSearchCV进行参数调优
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("Best parameters found: ", grid_search.best_params_)
# 使用最佳参数进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 评估模型性能
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
print("Mean Absolute Error for each time interval: ", mae)
# 可视化预测结果
time_intervals = ['15 min', '30 min', '45 min', '60 min']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, ax in enumerate(axes.flat):
    ax.plot(y_test.iloc[:, i].values, label='True Values')
    ax.plot(y_pred[:, i], label='Predicted Values')
    ax.set_title(f'CGM Prediction for {time_intervals[i]}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('CGM (mg / dl)')
    ax.legend()

plt.tight_layout()
plt.show()