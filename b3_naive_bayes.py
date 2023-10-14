from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import *

df = get_dataframe()

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Dự đoán trên tập huấn luyện và tập kiểm tra
y_pred_train = naive_bayes.predict(X_train)
y_pred_test = naive_bayes.predict(X_test)

# Tính ma trận nhầm lẫn cho tập huấn luyện và tập kiểm tra
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

print()
accuracy_for_train = np.round((cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)
accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)
print('Accuracy for training set for Naive Bayes = {}'.format(accuracy_for_train))
print('Accuracy for test set for Naive Bayes = {}'.format(accuracy_for_test))
