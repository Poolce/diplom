import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Генерация данных
data = np.random.rand(100, 5)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])

# Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Применение PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Коэффициенты главных компонент
components = pca.components_

# Умножение стандартизированных данных на матрицу компонентов
manual_transformed_data = np.dot(scaled_data, components.T)

# Проверка идентичности
print(np.allclose(principal_components, manual_transformed_data))
