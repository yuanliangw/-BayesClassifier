# -BayesClassifier
一次简单的贝叶斯分类实验
# 朴素贝叶斯分类器

这是一个用 Python 实现的朴素贝叶斯分类器，用于分类任务。它包括训练模型和预测类别的功能。

## 文件结构

- `naive_bayes.py`: 主要的分类器代码。
- `train1.txt`: 训练数据集文件。
- `predict1.txt`: 预测数据集文件。
- `预测的类标号_实现.txt`: 保存预测结果的文件。

## 依赖

该项目没有任何外部依赖，只需 Python 3 环境即可运行。

## 使用说明

### 1. 准备数据

确保 `train1.txt` 和 `predict1.txt` 文件在当前目录中，并按以下格式组织数据：

`train1.txt` 文件格式：

特征1 特征2 ... 特征n 标签
特征1 特征2 ... 特征n 标签
...


`predict1.txt` 文件格式：

特征1 特征2 ... 特征n
特征1 特征2 ... 特征n
...


### 2. 运行代码

在命令行中运行以下命令：

```bash
python naive_bayes.py
3. 查看结果
运行代码后，预测结果将保存到 预测的类标号_实现.txt 文件中，每行对应一个预测的类标签。

代码说明
NaiveBayesClassifier 类
class NaiveBayesClassifier:
    def __init__(self):
        self.prior = defaultdict(float)
        self.cond_prob = defaultdict(lambda: defaultdict(lambda: 1))

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        classes = set(y)
        laplace = 1

        for c in classes:
            self.prior[c] = (sum(1 for label in y if label == c) + laplace) / (n_samples + len(classes) * laplace)
        
        for i in range(n_features):
            feature_values = {x[i] for x in X}
            for c in classes:
                feature_counts = defaultdict(int)
                for index, label in enumerate(y):
                    if label == c:
                        feature_counts[X[index][i]] += 1
                for value in feature_values:
                    self.cond_prob[c][(i, value)] = (feature_counts[value] + laplace) / (sum(feature_counts.values()) + len(feature_values) * laplace)

    def predict(self, X):
        predictions = []
        for sample in X:
            max_prob = -1
            best_class = None
            for c, prior_prob in self.prior.items():
                posterior_prob = prior_prob
                for i, value in enumerate(sample):
                    posterior_prob *= self.cond_prob[c][(i, value)]
                if posterior_prob > max_prob:
                    max_prob = posterior_prob
                    best_class = c
            predictions.append(best_class)
        return predictions
加载数据函数
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
保存预测结果函数
def save_predictions(predictions, file_path):
    with open(file_path, 'w') as file):
        for pred in predictions:
            file.write(f"{pred}\n")
主程序
X_train, y_train = load_data('train1.txt')
X_test, _ = load_data('predict1.txt')

model = NaiveBayesClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("预测的类标号为:", predictions)

save_predictions(predictions, "预测的类标号_实现.txt")
print("预测结果已保存至 '预测的类标号_实现.txt'。")
