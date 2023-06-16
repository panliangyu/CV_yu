import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset_path = {'0': r'D:\Ubuntu\ABC_DATA\0_9\data_0',
                '1': r'D:\Ubuntu\ABC_DATA\0_9\data_1',
                '2': r'D:\Ubuntu\ABC_DATA\0_9\data_2',
                '3': r'D:\Ubuntu\ABC_DATA\0_9\data_3',
                '4': r'D:\Ubuntu\ABC_DATA\0_9\data_4',
                '5': r'D:\Ubuntu\ABC_DATA\0_9\data_5',
                '6': r'D:\Ubuntu\ABC_DATA\0_9\data_6',
                '7': r'D:\Ubuntu\ABC_DATA\0_9\data_7',
                '8': r'D:\Ubuntu\ABC_DATA\0_9\data_8',
                '9': r'D:\Ubuntu\ABC_DATA\0_9\data_9'}

X, y = [], []
for label, path in dataset_path.items():
    for f in os.listdir(path):
        if f.endswith('.png'):
            img = cv2.imread(os.path.join(path, f))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (20, 20))
            X.append(resized.reshape(-1))
            y.append(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

with open('knn_data.pkl', 'wb') as f:
    pickle.dump(knn, f, protocol=2)




