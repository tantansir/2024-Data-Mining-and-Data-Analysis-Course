# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 23:33:41 2024

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict

# 定义数据文件夹路径
data_folder = 'T2DM'

# 特征选择
features = ['Insulin dose - s.c.', 'Non-insulin hypoglycemic agents', 'CSII - bolus insulin (Novolin R, IU)',
            'Insulin dose - i.v.', 'take_food', 'Age (years)', 'Height (m)', 'Weight (kg)', 'BMI (kg/m2)',
            'Smoking History (pack year)', 'Alcohol Drinking History (drinker/non-drinker)', 'Type of Diabetes',
            'Duration of diabetes (years)', 'Acute Diabetic Complications', 'Hypoglycemia (yes/no)', 'Gender (Female=0, Male=1)',
            'hour', 'CGM_lag1', 'CGM_lag2']

# 定义创建滞后特征的函数
def create_lagged_features(df, target_col, lags):
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(-lag)
    return df

# 读取文件并按病人分组
patient_data = defaultdict(list)

for file in os.listdir(data_folder):
    if file.endswith('.xlsx'):
        patient_id = '_'.join(file.split('_')[:2])  # 假设前两个部分表示病人ID
        file_path = os.path.join(data_folder, file)
        df = pd.read_excel(file_path)
        patient_data[patient_id].append(df)

# 初始化记录MAE和MSE的列表
all_mae = []
all_mse = []

# 遍历每个病人的数据
for patient_id, dfs in patient_data.items():
    # 合并属于同一个病人的所有表格
    df = pd.concat(dfs, ignore_index=True)
    
    # 转换日期时间特征
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M:%S')
    df['hour'] = df['Date'].dt.hour

    # 创建滞后特征
    df = create_lagged_features(df, 'CGM (mg / dl)', lags=[15, 30, 45, 60])
    df.dropna(inplace=True)
    
    # 分割数据
    X_file = df[features]
    y_file = df[['CGM (mg / dl)_lag15', 'CGM (mg / dl)_lag30', 'CGM (mg / dl)_lag45', 'CGM (mg / dl)_lag60']]
    
    X_train = X_file.iloc[:-4, :]
    y_train = y_file.iloc[:-4, :]
    
    X_test = X_file.iloc[-4:, :]
    y_test = y_file.iloc[-4:, :]
    
    # 处理分类特征（如果有）
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # 确保训练集和测试集具有相同的列
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    # 创建预处理和模型管道
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), X_train.columns)
    ])

    # 创建SVM回归模型，使用给定的最佳参数
    svm = SVR(kernel='sigmoid', C=0.1, epsilon=0.1)

    # 创建多输出回归器
    multi_output_svr = MultiOutputRegressor(svm)

    # 将预处理器和模型组成管道
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', multi_output_svr)
    ])
    
    # 训练模型
    pipeline.fit(X_train, y_train)

    # 使用最佳参数进行预测
    y_pred = pipeline.predict(X_test)

    # 评估模型性能
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    all_mae.append(mae)
    all_mse.append(mse)
    print(f"Mean Absolute Error for each time interval for patient {patient_id}: ", mae)
    print(f"Mean Squared Error for each time interval for patient {patient_id}: ", mse)

# 打印所有病人的平均MAE和MSE
print("Average Mean Absolute Error for all patients: ", np.mean(all_mae, axis=0))
print("Average Mean Squared Error for all patients: ", np.mean(all_mse, axis=0))
