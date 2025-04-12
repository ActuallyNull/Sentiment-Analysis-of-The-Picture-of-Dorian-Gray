from sentimentAnalysis import *
import matplotlib.pyplot as plt
import numpy as np
from pairPlots import df_pos_nrc

# Calculate correlation matrix
corr = df_pos_nrc.corr()

# Set up figure
f = plt.figure(figsize=(19, 15))
fignum = f.number
plt.matshow(corr, fignum=f.number, cmap='plasma')  # optional: set a colormap like 'coolwarm'

# Set ticks and labels
plt.xticks(range(corr.shape[1]), corr.columns, fontsize=20, rotation=45)
plt.yticks(range(corr.shape[0]), corr.columns, fontsize=20)

# Add correlation numbers
for (i, j), val in np.ndenumerate(corr.values):
    plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=14, color='black')

# Colorbar
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)

# Title and save
plt.title('Correlation Matrix for Emotions in The Portrait of Dorian Gray', fontsize=40)
plt.tight_layout()
plt.savefig("plots/correlation_matrix.png", format='png', bbox_inches='tight')
plt.show()
