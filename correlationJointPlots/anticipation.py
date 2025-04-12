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
output_dir = "plots/correlationJointPlots/anticipation"
os.makedirs(output_dir, exist_ok=True)

# Bar plot
fig = plt.figure(figsize=(20, 8))
sns.barplot(data=df_pos,
            x='Chapter_Name',
            y='anticipation_n',
            palette='plasma')
plt.title('Anticipation Score per Chapter', fontsize=26, pad=20)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Anticipation Score', fontsize=30)
plt.xlabel('')
plt.tight_layout()
plt.savefig(f"{output_dir}/anticipation_bar.png", format='png', bbox_inches='tight')
plt.close()

# Regression Plot 1: Anticipation vs Joy
r, _ = stats.pearsonr(df_pos['anticipation_n'], df_pos['joy_n'])
a0 = sns.jointplot(x="anticipation_n", y="joy_n", data=df_pos, kind='reg', fit_reg=True, ci=90, color="darkturquoise")
a0.fig.suptitle('Anticipation vs Joy (Regression)', fontsize=16)
a0.fig.subplots_adjust(top=0.95)
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                     xy=(0.1, 0.9), xycoords='axes fraction',
                     ha='left', va='center',
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
a0.fig.savefig(f"{output_dir}/anticipation_vs_joy_reg.png", bbox_inches='tight')
plt.close(a0.fig)

# KDE Plot 1
kde1 = sns.jointplot(x="anticipation_n", y="joy_n", data=df_pos, kind='kde', shade=True, cmap="rainbow")
kde1.fig.suptitle('Anticipation vs Joy (KDE)', fontsize=16)
kde1.fig.subplots_adjust(top=0.95)
kde1.fig.savefig(f"{output_dir}/anticipation_vs_joy_kde.png", bbox_inches='tight')
plt.close(kde1.fig)

# Regression Plot 2: Anticipation vs Disgust
r, _ = stats.pearsonr(df_pos['anticipation_n'], df_pos['disgust_n'])
a2 = sns.jointplot(x="anticipation_n", y="disgust_n", data=df_pos, kind='reg', color="darkturquoise")
a2.fig.suptitle('Anticipation vs Disgust (Regression)', fontsize=16)
a2.fig.subplots_adjust(top=0.95)
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                     xy=(0.1, 0.9), xycoords='axes fraction',
                     ha='left', va='center',
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
a2.fig.savefig(f"{output_dir}/anticipation_vs_disgust_reg.png", bbox_inches='tight')
plt.close(a2.fig)

# KDE Plot 2
kde2 = sns.jointplot(x="anticipation_n", y="disgust_n", data=df_pos, kind='kde', shade=True, cmap="rainbow")
kde2.fig.suptitle('Anticipation vs Disgust (KDE)', fontsize=16)
kde2.fig.subplots_adjust(top=0.95)
kde2.fig.savefig(f"{output_dir}/anticipation_vs_disgust_kde.png", bbox_inches='tight')
plt.close(kde2.fig)
