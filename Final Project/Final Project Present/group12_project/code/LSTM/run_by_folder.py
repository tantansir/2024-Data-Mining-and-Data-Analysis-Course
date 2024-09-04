import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import Callback
from keras import backend as K
import matplotlib.pyplot as plt

# 读取T1DM文件夹中的所有Excel文件
folder_path = 'T1DM'
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# 定义病人的特征
patient_features = [
    'Age (years)', 'Height (m)', 'Weight (kg)', 'BMI (kg/m2)',
    'Smoking History (pack year)', 'Alcohol Drinking History (drinker/non-drinker)',
    'Type of Diabetes', 'Duration of Diabetes (years)', 'Acute Diabetic Complications',
    'Hypoglycemia (yes/no)', 'Gender (Female=0, Male=1)'
]

# 初始化LabelEncoder
label_encoders = {feature: LabelEncoder() for feature in patient_features}

# 自定义回调函数以记录每个周期的指标
class MetricsHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss')}, val_loss = {logs.get('val_loss')}, mse = {logs.get('mse')}, mae = {logs.get('mae')}, acc = {logs.get('acc')}")

# 自定义准确度度量函数
def custom_accuracy(y_true, y_pred):
    threshold = 0.1  # 误差阈值，可以根据需求调整
    diff = K.abs(y_true - y_pred)
    correct = K.less(diff, threshold)
    return K.mean(K.cast(correct, K.floatx()))

# 遍历每个文件
for file in file_list:
    # 读取数据
    file_path = os.path.join(folder_path, file)
    data = pd.read_excel(file_path)

    # 读取每个文件的行数
    num_rows = data.shape[0]
    print(f"{file} has {num_rows} rows")

    # 提取病人的特征并进行编码，只读取第一行
    patient_data = data[patient_features].iloc[0].to_frame().T
    for feature in patient_features:
        if patient_data[feature].dtype == 'object':
            patient_data[feature] = label_encoders[feature].fit_transform(patient_data[feature])

    # 选择血糖记录特征
    cgm_features = [
        'CGM (mg / dl)', 'Insulin dose - s.c.', 'Non-insulin hypoglycemic agents',
        'CSII - bolus insulin (Novolin R, IU)', 'Insulin dose - i.v.',
        'take_food', 'Smoking History (pack year)',
        'Alcohol Drinking History (drinker/non-drinker)'
    ]

    # 提取特征数据
    feature_data = data[cgm_features].copy()

    # 只对CGM数据进行归一化
    cgm_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_data['CGM (mg / dl)'] = cgm_scaler.fit_transform(feature_data[['CGM (mg / dl)']])
    # 创建包含病人特征的数据框
    patient_feature_data = pd.concat([feature_data] * len(feature_data), ignore_index=True)
    for feature in patient_features:
        patient_feature_data[feature] = patient_data[feature].values[0]

    # 创建输入特征和目标值
    sequence_length = 10  # 使用前十个时间步进行预测
    X, y = [], []
    for i in range(len(patient_feature_data) - sequence_length):
        X.append(patient_feature_data.iloc[i:i + sequence_length].values)
        y.append(patient_feature_data.iloc[i + sequence_length]['CGM (mg / dl)'])  # 目标值是CGM (mg / dl)

    X = np.array(X)
    y = np.array(y)

    # 将数据拆分为训练集和测试集，最后四行为验证集
    X_train, X_val = X[:-4], X[-4:]
    y_train, y_val = y[:-4], y[-4:]

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))  # 添加Dropout层以防止过拟合
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # 添加Dropout层以防止过拟合
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', custom_accuracy])

    # 初始化自定义回调
    metrics_history = MetricsHistory()

    # 训练模型
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), callbacks=[metrics_history])

    # 预测验证集
    y_pred_scaled = model.predict(X_val)

    # 反归一化预测值和真实值
    y_pred = cgm_scaler.inverse_transform(y_pred_scaled)
    y_val = cgm_scaler.inverse_transform(y_val.reshape(-1, 1))

    # 计算均绝对误差（MAE）和均方误差（MSE）
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    acc = np.mean(np.abs(y_val - y_pred) < 0.1)  # 自定义准确度

    print(f'Mean Absolute Error for {file}: {mae}')
    print(f'Mean Squared Error for {file}: {mse}')
    print(f'Accuracy for {file}: {acc}')

    # 可视化预测值和真实值
    plt.figure(figsize=(12, 6))
    plt.plot(y_val, color='blue', label='Actual CGM values')
    plt.plot(y_pred, color='red', label='Predicted CGM values')
    plt.title(f'Actual vs Predicted CGM values for {file}')
    plt.xlabel('Time step')
    plt.ylabel('CGM (mg/dl)')
    plt.legend()
    plt.show()

    # 预测最后四行的血糖值
    last_sequence = patient_feature_data.iloc[-sequence_length:].values
    last_sequence = last_sequence.reshape((1, sequence_length, patient_feature_data.shape[1]))  # 调整输入形状
    predictions = []

    for _ in range(4):
        prediction = model.predict(last_sequence)
        predictions.append(prediction[0, 0])  # 从预测结果中提取单一值
        # 创建一个新的输入序列，将预测值与原始特征数据结合
        new_cgm = prediction.reshape((1, 1, 1))  # 预测值新形状
        new_features = last_sequence[:, -1, 1:].reshape((1, 1, -1))  # 其他特征保持不变
        new_sequence = np.concatenate((last_sequence[:, 1:, :], np.concatenate((new_cgm, new_features), axis=2)),
                                      axis=1)
        last_sequence = new_sequence

    # 反归一化预测值
    predictions = cgm_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # 输出预测结果
    print(f"Predicted CGM values for the last four entries for {file}:")
    for i, pred in enumerate(predictions, start=1):
        print(f"Prediction {i}: {pred}")