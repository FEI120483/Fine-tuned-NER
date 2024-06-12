import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('output/training_results.csv')

# 比较不同训练策略训练时间 & Macro F1
sns.scatterplot(data, y='Train Time', x='Macro_f1',hue='Tuning Tactics')
plt.show()