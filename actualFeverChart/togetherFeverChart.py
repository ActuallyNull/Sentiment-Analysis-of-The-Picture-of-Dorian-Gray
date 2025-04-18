import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

# Chapter numbers (x-axis)
chapters = list(range(1, 21))

# Emotion scores - from most negative (1) to most positive (20)
emotion_scores = [
    20,  # Ch1: Enchantment
    16,  # Ch2: Naive
    19,  # Ch3: Facination
    18,  # Ch4: Infatuation
    14,  # Ch5: Concern
    17,  # Ch6: Devotion
    3,   # Ch7: Cruelty
    13,  # Ch8: Remorse
    10,  # Ch9: Defensive
    11,  # Ch10: Unease
    15,  # Ch11: Curiosity
    12,  # Ch12: Pride
    1,   # Ch13: Fury
    9,   # Ch14: Manipulative
    5,   # Ch15: Irritable
    4,   # Ch16: Frantic
    8,   # Ch17: Anxiety
    2,   # Ch18: Paranoia
    7,   # Ch19: Shame
    6    # Ch20: Disappointment
]

# Emotion labels
emotion_labels = [
    "Enchantment", "Naive", "Fascination", "Infatuation", 
    "Concern", "Devotion", "Cruelty", "Numbness", 
    "Defensive", "Suspicious", "Curiosity", "Pride", 
    "Fury", "Manipulative", "Irritable", "Frantic", 
    "Anxiety", "Paranoia", "Shame", "Disappointment"
]

# Create a mapping of unique scores to positions for the y-axis
# Invert the order for y-axis (now lowest score at bottom, highest at top)
unique_scores = sorted(list(set(emotion_scores))) # No longer reversed - natural order
emotion_positions = {score: i for i, score in enumerate(unique_scores)}

# Calculate the y position for each emotion based on its score
y_positions = [emotion_positions[score] for score in emotion_scores]

# Create the figure
plt.figure(figsize=(14, 10))

# Create segments for colored lines based on direction
points = np.array([chapters, y_positions]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Determine direction of each segment (positive = up, negative = down)
# Using y_positions rather than emotional scores because we're using the plot positions
directions = []
for i in range(len(y_positions)-1):
    # If next y-position is higher, it's going toward more positive emotions
    # (because we've inverted the axis)
    if y_positions[i+1] > y_positions[i]:
        directions.append(1)  # Green for positive direction
    else:
        directions.append(0)  # Red for negative direction

# Create a colormap with just two colors: red and green
cmap = ListedColormap(['red', 'green'])
norm = plt.Normalize(0, 1)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(np.array(directions))
lc.set_linewidth(2)

# Add the colored line segments to the plot
line = plt.gca().add_collection(lc)

# Identify special chapters to highlight in red
highlight_chapters = [3, 5, 12, 13, 14, 15, 16, 17, 18, 20]
# Convert to 0-indexed for array access
highlight_indices = [ch-1 for ch in highlight_chapters]

# Plot points for all chapters
plt.scatter(chapters, y_positions, s=80, color='blue', zorder=3)

# Override color for highlighted chapters
for idx in highlight_indices:
    plt.scatter(chapters[idx], y_positions[idx], s=80, color='red', zorder=4)

# Create the y-tick labels corresponding to emotions
y_tick_labels = []
for score in unique_scores:
    # Find the first emotion with this score
    for i, s in enumerate(emotion_scores):
        if s == score:
            y_tick_labels.append(emotion_labels[i])
            break

# Set the y-ticks to show emotions instead of numerical values
plt.yticks(range(len(unique_scores)), y_tick_labels)

# Add labels and title
plt.title('Fever Chart of "The Picture of Dorian Gray"', fontsize=16)
plt.xlabel('Chapter', fontsize=14)
plt.ylabel('Emotional States', fontsize=14)
plt.xticks(chapters, fontsize=10)

# Add a grid for better readability
plt.grid(True, alpha=0.3)

# Add chapter numbers as annotations to help trace the narrative journey
for i, ch in enumerate(chapters):
    # Make annotation red for highlighted chapters
    color = 'red' if ch in highlight_chapters else 'black'
    plt.annotate(f"Ch {ch}", (chapters[i], y_positions[i]),
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center',
                fontsize=9,
                color=color)

# Add colored bands to indicate emotional zones
# Find positions for each emotional zone
negative_positions = [pos for score, pos in emotion_positions.items() if score < 7]
neutral_positions = [pos for score, pos in emotion_positions.items() if 7 <= score < 14]
positive_positions = [pos for score, pos in emotion_positions.items() if score >= 14]

if negative_positions:
    plt.axhspan(min(negative_positions) - 0.5, max(negative_positions) + 0.5, 
                alpha=0.2, color='darkred', label='Highly Negative')
if neutral_positions:
    plt.axhspan(min(neutral_positions) - 0.5, max(neutral_positions) + 0.5, 
                alpha=0.2, color='lightgray', label='Mixed/Neutral')
if positive_positions:
    plt.axhspan(min(positive_positions) - 0.5, max(positive_positions) + 0.5, 
                alpha=0.2, color='skyblue', label='Positive')

# Add legend elements for line colors and point highlights
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', lw=2, label='Moving Toward Positive'),
    Line2D([0], [0], color='red', lw=2, label='Moving Toward Negative'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Chapters with Multiple Emotions'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Chapters with Single Emotions'),
]

# Add emotional zone indicators to legend
if negative_positions:
    legend_elements.append(plt.Rectangle((0,0), 1, 1, alpha=0.2, color='darkred', label='Highly Negative'))
if neutral_positions:
    legend_elements.append(plt.Rectangle((0,0), 1, 1, alpha=0.2, color='lightgray', label='Mixed/Neutral'))
if positive_positions:
    legend_elements.append(plt.Rectangle((0,0), 1, 1, alpha=0.2, color='skyblue', label='Positive'))

# Add a legend with all elements
plt.legend(handles=legend_elements, loc='upper right')

# Set proper y-axis limits to avoid cutting off points
plt.ylim(-0.5, len(unique_scores)-0.5)

# Save and show the plot
plt.tight_layout()
plt.savefig('actualFeverChart/dorian_gray_fever_chart.png', dpi=300)
plt.close()

# Print the data to verify
print("Chapter-by-Chapter Emotional States:")
print("-" * 40)
for i in range(len(chapters)):
    highlight = " (highlighted)" if chapters[i] in highlight_chapters else ""
    print(f"Chapter {chapters[i]}: {emotion_labels[i]} (Score: {emotion_scores[i]}){highlight}")