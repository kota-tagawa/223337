import matplotlib.pyplot as plt
import pandas as pd

# CSVファイルからデータを読み込む
data = pd.read_csv('output.csv')

# プロット
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], color='blue', label='Original Data', alpha=0.6)
plt.scatter(data['noisy_x'], data['noisy_y'], color='red', label='Noisy Data', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Original and Noisy Data')
plt.grid(True)

# グラフを保存する
plt.savefig('output_plot.png')
plt.show()
