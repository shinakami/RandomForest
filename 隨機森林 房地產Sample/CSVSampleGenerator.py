import pandas as pd
import numpy as np

# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 定義參數
num_samples = 1000
areas = np.random.randint(50, 250, num_samples)  # 面積：50到250平米
bedrooms = np.random.randint(1, 5, num_samples)  # 臥室數：1到5
bathrooms = np.random.randint(1, 4, num_samples)  # 浴室數：1到4
floors = np.random.randint(1, 3, num_samples)  # 樓層數：1到3
year_built = np.random.randint(1950, 2022, num_samples)  # 建造年份：1950到2021
locations = np.random.choice(['Downtown', 'Suburbs', 'Rural'], num_samples)  # 地點
prices = areas * 3000 + bedrooms * 50000 + bathrooms * 10000 + (2022 - year_built) * -1000 + np.random.randint(-50000, 50000, num_samples)

# 創建DataFrame
data = pd.DataFrame({
    'area': areas,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'floors': floors,
    'year_built': year_built,
    'location': locations,
    'price': prices
})

# 保存為CSV文件
data.to_csv('real_estate_data.csv', index=False)

# 顯示前幾行數據
print(data.head())
