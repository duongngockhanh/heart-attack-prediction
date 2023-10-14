import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

df = get_dataframe()

# Phân phối của biến target dựa trên biến age
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='age', hue='target')
plt.title('Phân phối của biến target dựa trên biến age')
plt.savefig("visualization/a1_target_age.png")