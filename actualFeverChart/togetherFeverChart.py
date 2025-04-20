import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# Chapter numbers (x-axis)
chapters = list(range(1, 21))

# Emotion scores - from most negative (1) to most positive (20)
emotion_scores = [
    20,  # Ch1: Enchantment
    16,  # Ch2: Naive
    19,  # Ch3: Fascination
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
    "Concern", "Devotion", "Cruelty", "Remorse", 
    "Defensive", "Unease", "Curiosity", "Pride", 
    "Fury", "Manipulative", "Irritable", "Frantic", 
    "Anxiety", "Paranoia", "Shame", "Disappointment"
]

# Invert y-axis: higher emotion scores (positive) at the bottom
unique_scores = sorted(list(set(emotion_scores)), reverse=True)
emotion_positions = {score: i for i, score in enumerate(unique_scores)}
y_positions = [emotion_positions[score] for score in emotion_scores]

# Create the figure
plt.figure(figsize=(14, 10))

# Create segments for colored lines based on direction
points = np.array([chapters, y_positions]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Determine direction of each segment (positive = up, negative = down)
directions = []
for i in range(len(y_positions)-1):
    if y_positions[i+1] > y_positions[i]:
        directions.append(0)  # Green = emotionally improving
    else:
        directions.append(1)  # Red = emotionally worsening

# Color map for line segments
cmap = ListedColormap(['red', 'green'])
norm = plt.Normalize(0, 1)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(np.array(directions))
lc.set_linewidth(2)

# Add line segments
plt.gca().add_collection(lc)

# Highlight specific chapters
highlight_chapters = [3, 5, 12, 13, 14, 15, 16, 17, 18, 20]
highlight_indices = [ch-1 for ch in highlight_chapters]

# Plot chapter points
plt.scatter(chapters, y_positions, s=80, color='blue', zorder=3)
for idx in highlight_indices:
    plt.scatter(chapters[idx], y_positions[idx], s=80, color='red', zorder=4)

# Y-axis labels
y_tick_labels = []
for score in unique_scores:
    for i, s in enumerate(emotion_scores):
        if s == score:
            y_tick_labels.append(emotion_labels[i])
            break

plt.yticks(range(len(unique_scores)), y_tick_labels)

# Emotional zones
negative_scores = [s for s in unique_scores if s <= 6]
neutral_scores = [s for s in unique_scores if 7 <= s <= 13]
positive_scores = [s for s in unique_scores if s >= 14]

negative_positions = [emotion_positions[s] for s in negative_scores]
neutral_positions = [emotion_positions[s] for s in neutral_scores]
positive_positions = [emotion_positions[s] for s in positive_scores]

if negative_positions:
    plt.axhspan(min(negative_positions) - 0.5, max(negative_positions) + 0.5,
                alpha=0.2, color='darkred', label='Highly Negative')
if neutral_positions:
    plt.axhspan(min(neutral_positions) - 0.5, max(neutral_positions) + 0.5,
                alpha=0.2, color='lightgray', label='Mixed/Neutral')
if positive_positions:
    plt.axhspan(min(positive_positions) - 0.5, max(positive_positions) + 0.5,
                alpha=0.2, color='skyblue', label='Positive')

# Add annotations for chapters
for i, ch in enumerate(chapters):
    color = 'red' if ch in highlight_chapters else 'black'
    plt.annotate(f"Ch {ch}", (chapters[i], y_positions[i]),
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center',
                 fontsize=9,
                 color=color)

# Labels and title
plt.title('Fever Chart of "The Picture of Dorian Gray"', fontsize=16)
plt.xlabel('Chapter', fontsize=14)
plt.ylabel('Emotional States', fontsize=14)
plt.xticks(chapters, fontsize=10)
plt.grid(True, alpha=0.3)

# Legend
legend_elements = [
    Line2D([0], [0], color='green', lw=2, label='Moving Toward Positive'),
    Line2D([0], [0], color='red', lw=2, label='Moving Toward Negative'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Chapters with Multiple Emotions'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Chapters with Single Emotions'),
]

if negative_positions:
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, alpha=0.2, color='darkred', label='Highly Negative'))
if neutral_positions:
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, alpha=0.2, color='lightgray', label='Mixed/Neutral'))
if positive_positions:
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, alpha=0.2, color='skyblue', label='Positive'))

plt.legend(handles=legend_elements, loc='upper left')

# Final layout and save
plt.ylim(-0.5, len(unique_scores)-0.5)
plt.tight_layout()
plt.savefig('actualFeverChart/dorian_gray_fever_chart.png', dpi=300)
plt.close()

# Print chapter breakdown
print("Chapter-by-Chapter Emotional States:")
print("-" * 40)
for i in range(len(chapters)):
    highlight = " (highlighted)" if chapters[i] in highlight_chapters else ""
    print(f"Chapter {chapters[i]}: {emotion_labels[i]} (Score: {emotion_scores[i]}){highlight}")
