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
output_dir = "plots/correlationJointPlots/negative"
os.makedirs(output_dir, exist_ok=True)

# Bar plot
fig = plt.figure(figsize=(20, 8))
sns.barplot(data=df_pos,
            x='Chapter_Name',
            y='negative_n',
            palette='plasma')
plt.title("Negative Score per Chapter", fontsize=24, pad=20)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Negative Score', fontsize=30)
plt.xlabel('')
plt.tight_layout()
plt.savefig(f"{output_dir}/negative_bar.png", format='png', bbox_inches='tight')
plt.close()

# Regression Plot 1: Negative vs Anger
r, _ = stats.pearsonr(df_pos['negative_n'], df_pos['anger_n'])
a0 = sns.jointplot(x="negative_n", y="anger_n", data=df_pos, kind='reg', ci=90, color='firebrick')
a0.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                     xy=(0.1, 0.9), xycoords='axes fraction',
                     ha='left', va='center',
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
a0.fig.suptitle("Negative vs Anger (Regression)", fontsize=16)
a0.fig.tight_layout()
a0.fig.subplots_adjust(top=0.95)
a0.fig.savefig(f"{output_dir}/negative_vs_anger_reg.png", bbox_inches='tight')
plt.close(a0.fig)

# KDE Plot 1
kde1 = sns.jointplot(x="negative_n", y="anger_n", data=df_pos, kind='kde', shade=True, cmap='PuRd')
kde1.fig.suptitle("Negative vs Anger (KDE)", fontsize=16)
kde1.fig.tight_layout()
kde1.fig.subplots_adjust(top=0.95)
kde1.fig.savefig(f"{output_dir}/negative_vs_anger_kde.png", bbox_inches='tight')
plt.close(kde1.fig)

# Regression Plot 2: Negative vs trust
r, _ = stats.pearsonr(df_pos['negative_n'], df_pos['trust_n'])
a2 = sns.jointplot(x="negative_n", y="trust_n", data=df_pos, kind='reg', color='firebrick')
a2.ax_joint.annotate(f'$\\rho = {r:.3f}$',
                     xy=(0.1, 0.9), xycoords='axes fraction',
                     ha='left', va='center',
                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
a2.fig.suptitle("Negative vs Trust (Regression)", fontsize=16)
a2.fig.tight_layout()
a2.fig.subplots_adjust(top=0.95)
a2.fig.savefig(f"{output_dir}/negative_vs_trust_reg.png", bbox_inches='tight')
plt.close(a2.fig)

# KDE Plot 2
kde2 = sns.jointplot(x="negative_n", y="trust_n", data=df_pos, kind='kde', shade=True, cmap='PuRd')
kde2.fig.suptitle("Negative vs Trust (KDE)", fontsize=16)
kde2.fig.tight_layout()
kde2.fig.subplots_adjust(top=0.95)
kde2.fig.savefig(f"{output_dir}/negative_vs_trust_kde.png", bbox_inches='tight')
plt.close(kde2.fig)
