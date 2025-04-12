from sentimentAnalysis import *
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(30, 25))
# scatter plot
nRows = 2
nCols = 2
col_names = ['neg', 'neu', 'pos', 'compound']
axs_names = ['negative', 'neutral', 'positive', 'compound']

# Use consistent colormap
cmap = plt.cm.get_cmap('twilight_shifted')
slicedCM = cmap(np.linspace(0, 1, len(col_names)))

i = 0
j = 0
for val in range(len(col_names)):
    col_name = col_names[val]
    axs_name = axs_names[val]

    axs[i, j].plot(df_pos['Chapter_Name'],
                   df_pos[col_name],
                   marker='o',
                   markersize=12,
                   linewidth=4,
                   color=slicedCM[val])

    axs[i, j].set_xticks(df_pos['Chapter_Name'])
    axs[i, j].set_xticklabels(df_pos['Chapter_Name'], rotation=90)
    axs[i, j].set_yticks([df_pos[col_name].min(), df_pos[col_name].max()])
    axs[i, j].set_yticklabels(np.round([df_pos[col_name].min(), df_pos[col_name].max()], 2))
    axs[i, j].set_ylabel(axs_name, fontsize=30)
    axs[i, j].grid(True)
    axs[i, j].tick_params(axis='y', labelsize=20)
    axs[i, j].tick_params(axis='x', labelsize=20)
    

    if i < nRows - 1:
        i += 1
    else:
        i = 0
        j += 1
plt.tight_layout()
plt.suptitle("Sentiment Trends Across Chapters", fontsize=40, y=1.02)
plt.savefig("plots/neg_pos_neut_trends.png", format='png', bbox_inches='tight')