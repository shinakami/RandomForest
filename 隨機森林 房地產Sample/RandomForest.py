import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 假設我們有一個數據集，讀入數據
data = pd.read_csv('real_estate_data.csv')

# 將類別變量轉換為數值變量（例如，使用one-hot編碼）
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# 分割數據集為訓練集和測試集
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立隨機森林回歸模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 進行預測
y_pred = model.predict(X_test)

# 評估模型
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'均方根誤差（RMSE）: {rmse}')


# 獲取特徵重要性
feature_importances = model.feature_importances_

# 將特徵名和重要性組合成DataFrame
features = X.columns

feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# 按重要性排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 顯示前3個最重要的特徵
print(feature_importance_df.head(3))

# 可視化特徵重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.savefig("Feature Importance")
plt.show()
