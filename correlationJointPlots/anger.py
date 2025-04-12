import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from scipy import stats

# Access sentimentAnalysis.py one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentimentAnalysis import df_pos

# Create output directory if it doesn't exist
output_dir = "plots/correlationJointPlots"
os.makedirs(output_dir, exist_ok=True)

# Bar plot
fig = plt.figure(figsize=(20, 8))
sns.barplot(data=df_pos,
            x='Chapter_Name',
            y='anger_n',
            palette='plasma')
plt.title('Anger Score per Chapter', fontsize=26, pad=20)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Anger Score', fontsize=30)
plt.xlabel('')
plt.tight_layout()
plt.savefig(f"{output_dir}/anger/anger_bar.png", format='png', bbox_inches='tight')
plt.close()

# Regression Plot 1: Anger vs Negative
r, _ = stats.pearsonr(df_pos['anger_n'], df_pos['negative_n'])
a0 = sns.jointplot(x="anger_n", y="negative_n", data=df_pos, kind='reg', fit_reg=True, ci=90, color='red')
a0.fig.suptitle('Anger vs Negative (Regression)', fontsize=16)
a0.fig.subplots_adjust(top=0.95)  # Adjust to not cut off the title
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                     xy=(0.1, 0.9), xycoords='axes fraction',
                     ha='left', va='center',
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
a0.fig.savefig(f"{output_dir}/anger/anger_vs_negative_reg.png", bbox_inches='tight')
plt.close(a0.fig)

# KDE Plot 1
kde1 = sns.jointplot(x="anger_n", y="negative_n", data=df_pos, kind='kde', shade=True, cmap="Reds")
kde1.fig.suptitle('Anger vs Negative (KDE)', fontsize=16)
kde1.fig.subplots_adjust(top=0.95)
kde1.fig.savefig(f"{output_dir}/anger/anger_vs_negative_kde.png", bbox_inches='tight')
plt.close(kde1.fig)

# Regression Plot 2: Anger vs Disgust
r, _ = stats.pearsonr(df_pos['anger_n'], df_pos['disgust_n'])
a2 = sns.jointplot(x="anger_n", y="disgust_n", data=df_pos, kind='reg', color='red')
a2.fig.suptitle('Anger vs Disgust (Regression)', fontsize=16)
a2.fig.subplots_adjust(top=0.95)
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                     xy=(0.1, 0.9), xycoords='axes fraction',
                     ha='left', va='center',
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
a2.fig.savefig(f"{output_dir}/anger/anger_vs_disgust_reg.png", bbox_inches='tight')
plt.close(a2.fig)

# KDE Plot 2
kde2 = sns.jointplot(x="anger_n", y="disgust_n", data=df_pos, kind='kde', shade=True, cmap="Reds")
kde2.fig.suptitle('Anger vs Disgust (KDE)', fontsize=16)
kde2.fig.subplots_adjust(top=0.95)
kde2.fig.savefig(f"{output_dir}/anger/anger_vs_disgust_kde.png", bbox_inches='tight')
plt.close(kde2.fig)
