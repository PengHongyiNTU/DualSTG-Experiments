import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
no_fs_results = pd.read_csv('no_fs_results.csv')
gini_half_results = pd.read_csv('gini_half_results.csv')
stg_results = pd.read_csv('stg_results.csv')
two_step_results = pd.read_csv('two_step_results.csv')
sns.lineplot(data=no_fs_results, x="# features", y="accuracy", 
label='No Feature Selection', marker='o', alpha=0.3)
sns.lineplot(data=gini_half_results, x="# features", y="accuracy",
label='Gini-Filter Half 0.5', marker='o', alpha=0.3)
sns.lineplot(data=stg_results, x="# features", y="accuracy", 
label='STG', marker='s')
sns.lineplot(data=two_step_results, x="# features", y="accuracy",
label='GINI + STG', marker='o')
plt.xlabel('Number of Noisy Features')
plt.ylabel('Test Accuracy')
plt.show()