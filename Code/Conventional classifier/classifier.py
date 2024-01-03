import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

# 1. 读取数据
def read_excel_files(filenames):
    dataframes = [pd.read_excel(filename, header=0) for filename in filenames]
    return dataframes

# 2. 填充最大长度操作
def pad_dataframes(dfs):
    # 找到所有文件中最大的行数
    max_rows = max(df.shape[0] for df in dfs)
    print(max_rows)
    # 对每个文件进行填充
    for i in range(len(dfs)):
        # 计算需要填充的行数
        num_rows_to_add = max_rows - dfs[i].shape[0]
        
        # 创建一个和dfs[i]相同形状的0 dataframe，行数为num_rows_to_add
        df_to_add = pd.DataFrame(0, index=range(num_rows_to_add), columns=dfs[i].columns)
        
        # 将df_to_add添加到dfs[i]的末尾
        dfs[i] = pd.concat([dfs[i], df_to_add], axis=0)
        
    return dfs

# 2. 融合特征
def merge_features(intercept, pixel, slope, weights):
    # 此处是一个简单的融合方式，你可以根据需要进行修改
    return np.asarray(intercept) * weights[0] + np.asarray(pixel) * weights[1] + np.asarray(slope) * weights[2]

def merge_features_for_classes(dfs, weights):
    # 前提是dfs[0]和dfs[1]是一组数据，dfs[2]和dfs[3]是第二组，以此类推。
    features_nz = merge_features(dfs[0], dfs[2], dfs[4], weights)
    features_tc = merge_features(dfs[1], dfs[3], dfs[5], weights)
    # print(features_nz.shape,features_tc.shape)
    # 使用ignore_index=True来忽略原始的列名
    return np.vstack((features_nz.T, features_tc.T))

# 3. 数据预处理
def preprocess_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# 4. 训练模型并进行预测
def train_and_predict(models, X_train, X_test, y_train):
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
    return predictions

# 5. 评估和可视化
def plot_evaluation_metrics(models, results):
    # 设置图形大小
    plt.figure(figsize=(14, 10))

    # 指标名称
    metrics = ["accuracy", "auc", "recall", "precision"]

    # 每个模型的指标值
    bar_width = 0.2
    index = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        bars = plt.bar(index + i * bar_width, values, bar_width, label=metric)
        
        # 在每个柱子上添加数值
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

    # 设置标题和标签
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.xticks(index + bar_width, models)
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_and_plot(models, predictions, y_test):
    results = {}
    for name, pred in predictions.items():
        accuracy = accuracy_score(y_test, pred)
        auc_score = roc_auc_score(y_test, pred)
        recall = recall_score(y_test, pred)
        precision = precision_score(y_test, pred)
        
        results[name] = {
            "accuracy": accuracy,
            "auc": auc_score,
            "recall": recall,
            "precision": precision
        }
        
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print("---------------------------")
    
    # 绘制评估指标的柱状图
    model_names = list(models.keys())
    plot_evaluation_metrics(model_names, results)
    
    return results

if __name__ == "__main__":
    # 读取数据
    files = ['data/intercept-nz.xlsx', 'data/intercept-tc.xlsx', 'data/pixel-nz.xlsx',
             'data/pixel-tc.xlsx', 'data/slope-nz.xlsx', 'data/slope-tc.xlsx']
 
    dfs = read_excel_files(files)

    # 对所有数据进行填充操作
    dfs = pad_dataframes(dfs)

    # 为每个类别融合特征
    features = merge_features_for_classes(dfs, [0.7, 0.2, 0.1])  # 修改weights来调整融合的权重

    # 对于每个特征，需要按照nz和tc来获得标签
    labels = [0] * dfs[0].shape[1] + [1] * dfs[1].shape[1]  # 假设nz为0，tc为1

    print("合并后的数据形状：",features.shape)

    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data(features, labels)

    # 使用0填充NaN值
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # 定义模型
    models = {
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier()
    }

    # 训练模型并进行预测
    predictions = train_and_predict(models, X_train, X_test, y_train)

    # 评估和可视化
    results = evaluate_and_plot(models, predictions, y_test)
