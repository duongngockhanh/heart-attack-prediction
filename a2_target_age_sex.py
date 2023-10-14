import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

df = get_dataframe()

# Biểu đồ barplot của độ tuổi vs giới tính với hue=target
plt.figure(figsize=(12, 6))
sns.barplot(y='age', x='sex', hue='target', data=df)
plt.title('Mối quan hệ giữa độ tuổi, giới tính và khả năng bị bệnh tim')
plt.savefig("visualization/a2_target_age_sex.png")