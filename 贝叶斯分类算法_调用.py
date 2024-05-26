from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 读取训练和测试数据
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file.readlines()]
    return data
# 保存预测结果到文件
def save_predictions(predictions, file_path):
    with open(file_path, 'w') as file:
        for pred in predictions:
            file.write(f"{pred}\n")
# 编码处理函数
def encode_features(data):
    encoders = {}
    encoded_data = np.empty((len(data), len(data[0])))
    for i in range(len(data[0])):  # 对每一个特征进行编码
        encoder = LabelEncoder()
        feature = [sample[i] for sample in data]
        encoded_data[:, i] = encoder.fit_transform(feature)
        encoders[i] = encoder
    return encoded_data, encoders
# 读取训练数据
train_data = read_data("train1.txt")
# 读取测试数据
test_data = read_data("predict1.txt")
# 准备训练数据
X_train = [d[:-1] for d in train_data]
y_train = [d[-1] for d in train_data]
# 编码训练数据
X_encoded, encoders = encode_features(X_train)
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y_train)
# 训练朴素贝叶斯分类器
model = CategoricalNB()
model.fit(X_encoded, y_encoded)
# 编码测试数据
X_test_encoded = np.array([[encoders[i].transform([x[i]])[0] for i in range(len(x))] for x in test_data])
# 进行预测
y_pred = model.predict(X_test_encoded)
predictions = y_encoder.inverse_transform(y_pred)
# 打印预测的类标号
print("预测的类标号为:", predictions)
# 保存预测结果
save_predictions(predictions, "预测的类标号_调用.txt")
print("预测结果已保存至 '预测的类标号_调用.txt'。")



