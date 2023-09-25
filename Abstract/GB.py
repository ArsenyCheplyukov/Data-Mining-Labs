import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

# Загрузим данные
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Посмотрим на размерности данных:
print("Размерность X_train:", X_train.shape)
print("Размерность X_test:", X_test.shape)
print("Размерность y_train:", y_train.shape)
print("Размерность y_test:", y_test.shape)

# посмотрим набор данных:
pd.DataFrame(X_train).head()

# Можно заметить, что все признаки имеют разный масштаб, поэтому перед обучением модели
# необходимо провести нормализацию данных. Для этого можно использовать стандартизацию:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Далее можно построить матрицу корреляций между признаками, чтобы оценить степень
# линейной зависимости между ними:

corr_matrix = np.corrcoef(X_train.T)
sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm", xticklabels=False, yticklabels=False
)
plt.show()

clf = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
).fit(X_train, y_train)
print(clf.score(X_test, y_test))
