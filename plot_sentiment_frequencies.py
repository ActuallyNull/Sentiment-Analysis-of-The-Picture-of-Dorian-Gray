from sentimentAnalysis import *
import matplotlib.pyplot as plt
import random
import numpy as np

fig, axs = plt.subplots(10, 1, figsize=(30, 40))

col_names = ['joy_n', 'positive_n', 'anticipation_n', 'sadness_n',
             'surprise_n', 'negative_n', 'anger_n', 'disgust_n', 'trust_n', 'fear_n']

cmap = plt.cm.get_cmap('twilight_shifted')
slicedCM = cmap(np.linspace(0, 1, len(col_names)))

for i, col_name in enumerate(col_names):
    axs[i].plot(df_pos['Chapter_Name'], 
                df_pos[col_name], 
                marker='o',  
                markersize=12, 
                linewidth=4,  
                color=slicedCM[i])

    # Set y ticks and labels
    yticks = [df_pos[col_name].min(), df_pos[col_name].max()]
    axs[i].set_yticks(yticks)
    axs[i].set_yticklabels(np.round(yticks, 2), fontsize=12)

    # Label y-axis
    axs[i].set_ylabel(col_name, fontsize=16)

    # Add grid
    axs[i].grid(True)

    # X-axis settings
    axs[i].set_xticks(range(len(df_pos['Chapter_Name'])))
    axs[i].set_xticklabels(df_pos['Chapter_Name'], rotation=90, fontsize=20)
fig.suptitle("Sentiment Trends Across Chapters", fontsize=40, y=1.02)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig("plots/sentiments_trends.png", format='png', bbox_inches='tight')