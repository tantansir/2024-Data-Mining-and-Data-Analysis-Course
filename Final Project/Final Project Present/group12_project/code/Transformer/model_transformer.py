import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader, TensorDataset

# 定义分类特征和数值特征
categorical_features = [
    'Insulin dose - s.c.', 'Non-insulin hypoglycemic agents',
    'CSII - bolus insulin (Novolin R, IU)', 'Insulin dose - i.v.', 'take_food',
    'Gender (Female=0, Male=1)', 'Alcohol Drinking History (drinker/non-drinker)',
    'Type of Diabetes', 'Acute Diabetic Complications', 'Hypoglycemia (yes/no)'
]

numerical_features = [
    'CGM (mg / dl)', 'Age (years)', 'Height (m)', 'Weight (kg)', 'BMI (kg/m2)',
    'Smoking History (pack year)', 'Duration of Diabetes (years)',
    'hour', 'CGM_lag1', 'CGM_lag2'
]


# 加载和预处理数据
def load_and_process_data(folder_paths):
    all_data = []

    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx') and not filename.startswith('~$'):
                file_path = os.path.join(folder_path, filename)
                data = pd.read_excel(file_path)
                all_data.append(data)

    return all_data


folder_paths = ['T1DM', 'T2DM']
all_data = load_and_process_data(folder_paths)

print('数据加载完成')
# 合并所有数据
data = pd.concat(all_data, ignore_index=True)

# 定义超参数
window_size = 12
embedding_size = 72
num_heads = 12
num_blocks = 8
dropout_rate = 0.1
batch_size = 16
learning_rate = 1e-5
shuffle = True


# 创建滑动窗口
def create_windows(features, target, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i - window_size:i][features].values)
        y.append(data.iloc[i][target])
    return np.array(X), np.array(y)


print(categorical_features)
print(numerical_features)

dynamic_categorical_features = [col for col in categorical_features if col in data.columns]
dynamic_numerical_features = [col for col in numerical_features if col in data.columns]

# 创建数据窗口
X_dynamic_cat, _ = create_windows(dynamic_categorical_features, 'CGM (mg / dl)', window_size)
X_dynamic_num, y = create_windows(dynamic_numerical_features, 'CGM (mg / dl)', window_size)

# 重新整形数据
X_dynamic_cat = X_dynamic_cat.reshape(-1, window_size, len(dynamic_categorical_features))
X_dynamic_num = X_dynamic_num.reshape(-1, window_size, len(dynamic_numerical_features))

# 数据集划分
X_train_cat, X_val_cat, X_train_num, X_val_num, y_train, y_val = train_test_split(
    X_dynamic_cat, X_dynamic_num, y, test_size=0.2, shuffle=shuffle)


# 定义模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_categorical_features, num_numerical_features, num_targets, embedding_size, num_heads,
                 num_blocks, dropout_rate):
        super(TimeSeriesTransformer, self).__init__()
        self.categorical_embedding = nn.Embedding(2, embedding_size)  # Assuming binary categorical features
        self.numerical_embedding = nn.Linear(num_numerical_features, embedding_size * num_numerical_features)

        combined_features = embedding_size * (num_categorical_features + num_numerical_features)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=combined_features, nhead=num_heads, dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_blocks)
        self.out = nn.Linear(combined_features, num_targets)

    def forward(self, x_cat, x_num):
        x_cat = self.categorical_embedding(x_cat.long())
        x_cat = x_cat.view(x_cat.shape[0], x_cat.shape[1], -1)

        x_num = self.numerical_embedding(x_num)
        x_num = x_num.view(x_num.shape[0], x_num.shape[1], -1)

        x = torch.cat((x_cat, x_num), dim=2)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        return self.out(x)


# 初始化模型和超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(
    num_categorical_features=len(dynamic_categorical_features),
    num_numerical_features=len(dynamic_numerical_features),
    num_targets=1,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_blocks=num_blocks,
    dropout_rate=dropout_rate
).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 转换数据为张量并移动到设备
X_train_cat_tensor = torch.tensor(X_train_cat, dtype=torch.float32).to(device)
X_train_num_tensor = torch.tensor(X_train_num, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val_cat_tensor = torch.tensor(X_val_cat, dtype=torch.float32).to(device)
X_val_num_tensor = torch.tensor(X_val_num, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# 创建Tensor数据集
train_dataset = TensorDataset(torch.tensor(X_train_cat, dtype=torch.float32),
                              torch.tensor(X_train_num, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=200):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x_cat, x_num)
            loss = criterion(outputs, y.unsqueeze(1))
            mae = l1_loss(outputs, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            outputs = model(X_val_cat_tensor, X_val_num_tensor)
            val_loss = criterion(outputs, y_val_tensor)
        val_losses.append(val_loss.item())

        if epoch % 2 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {mae.item():.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


print('开始训练')
# 训练模型
train_model(model, criterion, optimizer, scheduler)


# 测试模型
def test_model(model, data, categorical_features, numerical_features, window_size):
    model.eval()
    all_y_test = []
    all_y_pred = []
    mae_list = []

    with torch.no_grad():
        for df in data:

            X_test_cat, X_test_num, y_test = [], [], []
            for i in range(len(df) - window_size - 4, len(df) - window_size):
                X_test_cat.append(df.iloc[i:i+window_size][categorical_features].values)
                X_test_num.append(df.iloc[i:i + window_size][numerical_features].values)
                y_test.append(df.iloc[i + window_size]['CGM (mg / dl)'])

            X_test_cat = np.array(X_test_cat)
            X_test_num = np.array(X_test_num)
            y_test = np.array(y_test)

            X_test_cat_tensor = torch.tensor(X_test_cat, dtype=torch.float32).to(device)
            X_test_num_tensor = torch.tensor(X_test_num, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

            y_pred = model(X_test_cat_tensor, X_test_num_tensor)

            mae = l1_loss(y_pred, y_test_tensor)
            mae_list.append(mae.item())

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred.cpu().numpy())

            print("Predicted values: ", y_pred.cpu().numpy().flatten())
            print("Actual values: ", y_test)

    # 可视化预测和真实值
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(all_y_test)), all_y_test, marker='o', label='Actual', color='blue')
    plt.plot(range(len(all_y_pred)), all_y_pred, marker='x', label='Predicted', color='red')
    plt.title('Predicted vs Actual Blood Glucose Levels')
    plt.xlabel('Time Step')
    plt.ylabel('CGM (mg / dl)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 单变量分析：误差与特征值的关系
    for feature in dynamic_numerical_features:
        plt.figure(figsize=(12, 6))
        plt.scatter([x[0] for x in X_val_num[:, 0, dynamic_numerical_features.index(feature)]],
                    np.array(all_y_test) - np.array(all_y_pred))
        plt.xlabel(feature)
        plt.ylabel('Prediction Error')
        plt.title(f'Prediction Error vs {feature}')
        plt.grid(True)
        plt.show()

    # 多变量分析：两个特征的交互作用
    feature1 = 'Alcohol Drinking History (drinker/non-drinker)'
    feature2 = 'Smoking History (pack year)'

    interaction = np.array([x[0] for x in X_val_num[:, 0, dynamic_numerical_features.index(feature1)]]) * \
                  np.array([x[0] for x in X_val_num[:, 0, dynamic_numerical_features.index(feature2)]])
    plt.figure(figsize=(12, 6))
    plt.scatter(interaction, np.array(all_y_test) - np.array(all_y_pred))
    plt.xlabel(f'{feature1} * {feature2}')
    plt.ylabel('Prediction Error')
    plt.title(f'Prediction Error vs {feature1} * {feature2}')
    plt.grid(True)
    plt.show()

    return np.mean(mae_list)


# 进行测试并可视化结果
test_mae = test_model(model, all_data, dynamic_categorical_features, dynamic_numerical_features, window_size)
print(f"Test MAE: {test_mae:.4f}")
