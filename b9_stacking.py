from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import *

df = get_dataframe()

# Tách dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo các mô hình cơ bản
dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
xgb = XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=100)
gc = GradientBoostingClassifier(random_state=42)
svc = SVC(kernel='rbf', random_state=42)
ad = AdaBoostClassifier(random_state=42)

# Sử dụng dự đoán từ các mô hình cơ bản để tạo thành tập dữ liệu mới cho mô hình tổng hợp
base_learners = [dtc, rfc, knn, xgb, gc, svc, ad]
X_train_meta = np.zeros((len(y_train), len(base_learners)))

for i, model in enumerate(base_learners):
    model.fit(X_train, y_train)
    X_train_meta[:, i] = model.predict(X_train)

# Huấn luyện mô hình tổng hợp trên tập dữ liệu mới
meta_learner = RandomForestClassifier(random_state=42)
meta_learner.fit(X_train_meta, y_train)

# Sử dụng dự đoán từ các mô hình cơ bản trên tập kiểm tra để tạo thành tập dữ liệu mới cho mô hình tổng hợp
X_test_meta = np.zeros((len(y_test), len(base_learners)))

for i, model in enumerate(base_learners):
    X_test_meta[:, i] = model.predict(X_test)

# Dự đoán bằng mô hình tổng hợp
y_pred = meta_learner.predict(X_test_meta)

# Tính toán độ chính xác và ma trận nhầm lẫn cho cả tập kiểm tra và tập huấn luyện
cm_test = confusion_matrix(y_test, y_pred)
accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)

# Dự đoán trên tập huấn luyện
X_train_meta = np.zeros((len(y_train), len(base_learners)))

for i, model in enumerate(base_learners):
    X_train_meta[:, i] = model.predict(X_train)

y_train_pred = meta_learner.predict(X_train_meta)

cm_train = confusion_matrix(y_train, y_train_pred)
accuracy_for_train = np.round((cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)

print('Accuracy for training set for Stacking = {}'.format(accuracy_for_train))
print('Accuracy for test set for Stacking = {}'.format(accuracy_for_test))
