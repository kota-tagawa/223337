import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv("output.csv")

# グラフを描画
plt.figure(figsize=(12, 6))

# 平均の差の2乗のグラフ
plt.subplot(1, 2, 1)
plt.plot(df['N'], df['MeanDiffSquared'], marker='o', label='Mean Diff Squared')
plt.xlabel('Sample Size (N)')
plt.ylabel('Difference Squared')
plt.title('Mean Difference Squared vs Sample Size')
plt.xscale('log')  # 横軸を対数スケールに設定
plt.legend()

# 分散の差の2乗のグラフ
plt.subplot(1, 2, 2)
plt.plot(df['N'], df['VarianceDiffSquared'], marker='o', label='Variance Diff Squared')
plt.xlabel('Sample Size (N)')
plt.ylabel('Difference Squared')
plt.title('Variance Difference Squared vs Sample Size')
plt.xscale('log')  # 横軸を対数スケールに設定
plt.legend()

plt.tight_layout()

# グラフをファイルに保存
plt.savefig('output_plot.png')

# グラフを表示
plt.show()