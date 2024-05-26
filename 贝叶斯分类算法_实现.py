from collections import defaultdict

class BayesClassifier:
    def __init__(self):
        # 先验概率：每个类别的概率
        self.prior = defaultdict(float)
        # 条件概率：在类别 c 下，特征 (i, value) 的概率，默认值为 1（用于拉普拉斯平滑）
        self.cond_prob = defaultdict(lambda: defaultdict(lambda: 1))

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        classes = set(y)
        laplace = 1

        # 计算先验概率 P(c)
        for c in classes:
            # P(c) = (Nc + laplace) / (N + k * laplace)
            # 其中 Nc 是类别 c 的样本数量，N 是总样本数量，k 是类别数量
            self.prior[c] = (sum(1 for label in y if label == c) + laplace) / (n_samples + len(classes) * laplace)

        # 计算条件概率 P(x_i|c)
        for i in range(n_features):
            # 提取第 i 个特征的所有可能取值
            feature_values = {x[i] for x in X}
            for c in classes:
                feature_counts = defaultdict(int)
                for index, label in enumerate(y):
                    if label == c:
                        feature_counts[X[index][i]] += 1
                # 应用拉普拉斯平滑计算条件概率
                for value in feature_values:
                    self.cond_prob[c][(i, value)] = (feature_counts[value] + laplace) / (
                                sum(feature_counts.values()) + len(feature_values) * laplace)

    def predict(self, X):

        # predictions - 预测标签列表
        predictions = []
        for sample in X:
            # 用于存储最大后验概率
            max_prob = -1
            # 用于存储最大后验概率对应的类别
            best_class = None
            for c, prior_prob in self.prior.items():
                # 后验概率初始化为先验概率 P(c)
                posterior_prob = prior_prob
                for i, value in enumerate(sample):
                    # P(c|x) ∝ P(c) * P(x|c)
                    posterior_prob *= self.cond_prob[c][(i, value)]
                # 找到具有最大后验概率的类别
                if posterior_prob > max_prob:
                    max_prob = posterior_prob
                    best_class = c
            predictions.append(best_class)
        return predictions


# 读取训练和预测数据
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    X = []
    y = []
    for line in lines:
        parts = line.strip().split()
        X.append(parts[:-1])
        y.append(parts[-1])
    return X, y

# 保存预测结果到文件
def save_predictions(predictions, file_path):
    with open(file_path, 'w') as file:
        for pred in predictions:
            file.write(f"{pred}\n")

# 主程序
X_train, y_train = load_data('train1.txt')
X_test, _ = load_data('predict1.txt')

# 实例化并训练模型
model = BayesClassifier()
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)
# 打印预测的类标号
print("预测的类标号为:", predictions)
# 保存预测结果
save_predictions(predictions, "预测的类标号_实现.txt")
print("预测结果已保存至 '预测的类标号_实现.txt'。")
