import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# 生成一些模拟数据
# 这里我们假设输入维度为3，输出维度为6
X, y = make_regression(n_samples=1000, n_features=3, n_targets=6, noise=0.1, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 创建 SVR 模型，这里使用 RBF 核
svr = SVR(kernel='rbf')

# 使用 MultiOutputRegressor 来扩展 SVR 以支持多输出
multi_output_svr = MultiOutputRegressor(svr)

# 训练模型
multi_output_svr.fit(X_train, y_train)

# 使用模型进行预测
y_pred = multi_output_svr.predict(X_test)

# 将预测值转换回原始尺度
y_pred_original = scaler_y.inverse_transform(y_pred)

# 打印一些预测结果进行查看
print("预测的输出值（部分）:", y_pred_original[:5])





import joblib

# 保存训练好的模型
joblib.dump(multi_output_svr, 'multi_output_svr_model.pkl')

# 加载模型
loaded_model = joblib.load('multi_output_svr_model.pkl')

# 使用加载的模型进行预测
y_pred_loaded = loaded_model.predict(X_test)

# 将预测值转换回原始尺度
y_pred_original_loaded = scaler_y.inverse_transform(y_pred_loaded)

# 打印一些预测结果进行查看
print("加载的模型预测的输出值（部分）:", y_pred_original_loaded[:5])
