import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 读取数据集路径
dataset_path = {'A': r'D:\Ubuntu\ABC_DATA\ABC\Sample011',
                'B': r'D:\Ubuntu\ABC_DATA\ABC\Sample012',
                'C': r'D:\Ubuntu\ABC_DATA\ABC\Sample013'}

# 读取数据集图片
X, y = [], []                                              #创建数据集列表
for label, path in dataset_path.items():
    for f in os.listdir(path):                             #遍历数据集路径
        if f.endswith('.png'):                             #遍历所有png图片
            img = cv2.imread(os.path.join(path, f))        #读取每一张图片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #图片转为灰度图
            resized = cv2.resize(gray, (20, 20))           #图片尺寸转为20×20
            X.append(resized.reshape(-1))                  #转化为一维向量并保存到X列表
            y.append(label)                                #每张图片的标签保存到y列表

# 将数据集分为训练集和测试集
# test_size表示测试数据集的比例；random_state为偏差拟合因子
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 KNN 模型
k = 5                                      # knn算法邻居因子，表示预测时与最近的多少个数据进行分类判定
knn = KNeighborsClassifier(n_neighbors=k)  # 初始化knn分类器
knn.fit(X_train, y_train)                  # 训练knn模型并生成分类器

# 在测试集上进行模型评估
accuracy = knn.score(X_test, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# 运行程序模型将被生成保存，不保存模型无法被调用
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f, protocol=2)