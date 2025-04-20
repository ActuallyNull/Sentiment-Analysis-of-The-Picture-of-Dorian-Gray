import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

# Function to create an emotion progression plot for a single chapter with colors based on direction
def plot_chapter_emotion_progression(chapter_num, emotions, scores, descriptions=None):
    """
    Create a plot showing the emotional progression within a single chapter
    with emotions as y-ticks and colored lines based on direction
    
    Parameters:
    - chapter_num: Chapter number
    - emotions: List of emotions in chronological order
    - scores: List of emotional scores (1=most negative, 20=most positive)
    - descriptions: Optional list of description/quote for each emotion
    """
    # Number of emotional states in this chapter
    num_emotions = len(emotions)
    # Create step numbers (1, 2, 3, etc.)
    steps = list(range(1, num_emotions + 1))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a mapping of unique scores to vertical positions (natural order - low at bottom)
    unique_scores = sorted(list(set(scores)))
    emotion_positions = {score: i for i, score in enumerate(unique_scores)}
    
    # Calculate y positions for each emotion based on its score
    y_positions = [emotion_positions[score] for score in scores]
    
    # Create segments for colored lines based on direction
    points = np.array([steps, y_positions]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Determine direction of each segment (positive = up, negative = down)
    directions = []
    for i in range(len(y_positions)-1):
        if y_positions[i+1] > y_positions[i]:
            directions.append(1)  # Green for positive direction (toward more positive)
        else:
            directions.append(0)  # Red for negative direction (toward more negative)
    
    # Create a colormap with just two colors: red and green
    cmap = ListedColormap(['red', 'green'])
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(directions))
    lc.set_linewidth(2)
    
    # Add the colored line segments to the plot
    line = ax.add_collection(lc)
    
    # Plot points for all steps
    ax.scatter(steps, y_positions, s=80, color='red', zorder=3)
    
    # Add descriptions as annotations if provided
    if descriptions:
        for i, desc in enumerate(descriptions):
            if desc:
                ax.annotate(f'"{desc}"', 
                           (steps[i], y_positions[i]), 
                           xytext=(10, 0), 
                           textcoords="offset points", 
                           ha='left', 
                           fontsize=9,
                           style='italic',
                           wrap=True)
    
    # Also add emotion labels directly on the plot for clarity
    for i, emotion in enumerate(emotions):
        ax.annotate(emotion, 
                   (steps[i], y_positions[i]), 
                   xytext=(0, 10), 
                   textcoords="offset points",
                   ha='center', 
                   fontsize=11,
                   weight='bold')
    
    # Set custom y-ticks to display emotions
    y_tick_labels = []
    for score in unique_scores:
        # Find the first emotion with this score
        for i, s in enumerate(scores):
            if s == score:
                y_tick_labels.append(emotions[i])
                break
    
    # Remove duplicate emotions in y_tick_labels while preserving order
    seen = set()
    y_tick_labels = [x for x in y_tick_labels if not (x in seen or seen.add(x))]
    
    ax.set_yticks(range(len(unique_scores)))
    ax.set_yticklabels(y_tick_labels)
    
    # Add colored regions to indicate emotional zones
    # Get the positions corresponding to our score thresholds
    negative_positions = [pos for score, pos in emotion_positions.items() if score < 7]
    neutral_positions = [pos for score, pos in emotion_positions.items() if 7 <= score < 14]
    positive_positions = [pos for score, pos in emotion_positions.items() if score >= 14]
    
    if negative_positions:
        ax.axhspan(min(negative_positions) - 0.5, max(negative_positions) + 0.5, 
                  alpha=0.2, color='darkred', label='Highly Negative')
    if neutral_positions:
        ax.axhspan(min(neutral_positions) - 0.5, max(neutral_positions) + 0.5, 
                  alpha=0.2, color='lightgray', label='Mixed/Neutral')
    if positive_positions:
        ax.axhspan(min(positive_positions) - 0.5, max(positive_positions) + 0.5, 
                  alpha=0.2, color='skyblue', label='Positive')
    
    # Set titles and labels
    ax.set_title(f'Chapter {chapter_num} Fever Chart', fontsize=14)
    ax.set_xlabel('Progression through Chapter', fontsize=12)
    ax.set_ylabel('Emotional States', fontsize=12)
    
    # Set x-axis ticks to numerical progression
    ax.set_xticks(steps)
    ax.set_xticklabels([f'Step {i}' for i in steps])
    
    # Set proper y-axis limits to avoid cutting off points
    ax.set_ylim(-0.5, len(unique_scores)-0.5)
    
    # Add a grid
    ax.grid(True, alpha=0.3)
    
    # Add a legend for line colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Moving Toward Positive'),
        Line2D([0], [0], color='red', lw=2, label='Moving Toward Negative'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Emotional States')
    ]
    
    # Add emotional zone indicators to legend
    if negative_positions:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, alpha=0.2, color='darkred', label='Highly Negative'))
    if neutral_positions:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, alpha=0.2, color='lightgray', label='Mixed/Neutral'))
    if positive_positions:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, alpha=0.2, color='skyblue', label='Positive'))
    
    # Add a legend with all elements
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

# Now create plots for each multi-emotion chapter
# Using emotion labels from the first script

# Chapter 3: Fascination -> Defensive -> Curiosity
plot_chapter_emotion_progression(
    chapter_num=3,
    emotions=["Fascination", "Defensive", "Curiosity"],
    scores=[19, 9, 15]
)
plt.savefig('actualFeverChart/chapter3_emotions_updated.png')
plt.close()

# Chapter 5: Concern
plot_chapter_emotion_progression(
    chapter_num=5,
    emotions=["Infatuation", "Concern"],
    scores=[18, 14]
)
plt.savefig('actualFeverChart/chapter5_emotions_updated.png')
plt.close()

# Chapter 12: Pride
plot_chapter_emotion_progression(
    chapter_num=12,
    emotions=["Defensive", "Suspicious", "Irritable", "Pride"],
    scores=[9, 10, 5, 12]
)
plt.savefig('actualFeverChart/chapter12_emotions_updated.png') 
plt.close()

# Chapter 13: Fury
plot_chapter_emotion_progression(
    chapter_num=13,
    emotions=["Anxiety", "Pride", "Fury", "Cruelty", "Shame"],
    scores=[8, 12, 1, 3, 7]
)
plt.savefig('actualFeverChart/chapter13_emotions_updated.png')
plt.close()

# Chapter 14: Manipulative
plot_chapter_emotion_progression(
    chapter_num=14,
    emotions=["Disappointment", "Manipulative"],
    scores=[6, 9]
)
plt.savefig('actualFeverChart/chapter14_emotions_updated.png')
plt.close()

# Chapter 15: Irritable
plot_chapter_emotion_progression(
    chapter_num=15,
    emotions=["Unease", "Irritable", "Paranoia"],
    scores=[11, 5, 2]
)
plt.savefig('actualFeverChart/chapter15_emotions_updated.png')
plt.close()

# Chapter 16: Frantic
plot_chapter_emotion_progression(
    chapter_num=16,
    emotions=["Frantic", "Terror"],
    scores=[4, 2]
)
plt.savefig('actualFeverChart/chapter16_emotions_updated.png')
plt.close()

# Chapter 17: Anxiety
plot_chapter_emotion_progression(
    chapter_num=17,
    emotions=["Remorse", "Anxiety"],
    scores=[13, 8]
)
plt.savefig('actualFeverChart/chapter17_emotions_updated.png')
plt.close()

# Chapter 18: Paranoia
plot_chapter_emotion_progression(
    chapter_num=18,
    emotions=["Paranoia", "Shame"],
    scores=[2, 7]
)
plt.savefig('actualFeverChart/chapter18_emotions_updated.png')
plt.close()

# Chapter 20: Disappointment
plot_chapter_emotion_progression(
    chapter_num=20,
    emotions=["Naive", "Disappointment", "Shame"],
    scores=[16, 6, 7]
)
plt.savefig('actualFeverChart/chapter20_emotions_updated.png')
plt.close()

# Creating a figure with subplots for all chapter progressions
def create_all_chapter_progressions():
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))  # Adjust grid to fit 10 subplots
    axs = axs.flatten()
    
    # Data for each chapter - updated with emotions from first script
    chapter_data = [
        # Chapter 3
        {
            'chapter': 3,
            'emotions': ["Fascination", "Defensive", "Curiosity"],
            'scores': [19, 9, 15],
            'index': 0
        },
        # Chapter 5
        {
            'chapter': 5,
            'emotions': ["Infatuation", "Concern"],
            'scores': [18, 14],
            'index': 1
        },
        # Chapter 12
        {
            'chapter': 12,
            'emotions': ["Defensive", "Suspicious", "Irritable", "Pride"],
            'scores': [9, 10, 5, 12],
            'index': 2
        },
        # Chapter 13
        {
            'chapter': 13,
            'emotions': ["Anxiety", "Pride", "Fury", "Cruelty", "Shame"],
            'scores': [8, 12, 1, 3, 7],
            'index': 3
        },
        # Chapter 14
        {
            'chapter': 14,
            'emotions': ["Disappointment", "Manipulative"],
            'scores': [6, 9],
            'index': 4
        },
        # Chapter 15
        {
            'chapter': 15,
            'emotions': ["Unease", "Irritable", "Paranoia"],
            'scores': [11, 5, 2],
            'index': 5
        },
        # Chapter 16
        {
            'chapter': 16,
            'emotions': ["Frantic", "Paranoia"],
            'scores': [4, 2],
            'index': 6
        },
        # Chapter 17
        {
            'chapter': 17,
            'emotions': ["Remorse", "Anxiety"],
            'scores': [13, 8],
            'index': 7
        },
        # Chapter 18
        {
            'chapter': 18,
            'emotions': ["Paranoia", "Shame"], 
            'scores': [2, 7],
            'index': 8
        },
        # Chapter 20
        {
            'chapter': 20,
            'emotions': ["Naive", "Disappointment", "Shame"],
            'scores': [16, 6, 7],
            'index': 9
        }
    ]
    
    for data in chapter_data:
        i = data['index']
        chapter = data['chapter']
        emotions = data['emotions']
        scores = data['scores']
        steps = list(range(1, len(emotions) + 1))
        
        # Create mapping of emotions to y positions (based on scores)
        unique_scores = sorted(list(set(scores)))
        emotion_positions = {score: i for i, score in enumerate(unique_scores)}
        y_positions = [emotion_positions[score] for score in scores]
        
        # Create segments for colored lines
        if len(steps) > 1:  # Only if we have more than one point
            points = np.array([steps, y_positions]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Determine direction of each segment
            directions = []
            for j in range(len(y_positions)-1):
                if y_positions[j+1] > y_positions[j]:
                    directions.append(1)  # Green
                else:
                    directions.append(0)  # Red
            
            # Create colored line segments
            cmap = ListedColormap(['red', 'green'])
            norm = plt.Normalize(0, 1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(directions))
            lc.set_linewidth(2)
            
            # Add the colored line segments to the plot
            axs[i].add_collection(lc)
        
        # Plot points
        axs[i].scatter(steps, y_positions, s=60, color='red', zorder=3)
        
        # Set the y-ticks to show emotions
        y_tick_labels = []
        for score in unique_scores:
            for j, s in enumerate(scores):
                if s == score:
                    y_tick_labels.append(emotions[j])
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        y_tick_labels = [x for x in y_tick_labels if not (x in seen or seen.add(x))]
        
        axs[i].set_yticks(range(len(unique_scores)))
        axs[i].set_yticklabels(y_tick_labels, fontsize=8)
        
        # Add emotional zones
        negative_positions = [pos for score, pos in emotion_positions.items() if score < 7]
        neutral_positions = [pos for score, pos in emotion_positions.items() if 7 <= score < 14]
        positive_positions = [pos for score, pos in emotion_positions.items() if score >= 14]
        
        if negative_positions:
            axs[i].axhspan(min(negative_positions) - 0.5, max(negative_positions) + 0.5, 
                         alpha=0.2, color='darkred')
        if neutral_positions:
            axs[i].axhspan(min(neutral_positions) - 0.5, max(neutral_positions) + 0.5, 
                         alpha=0.2, color='lightgray')
        if positive_positions:
            axs[i].axhspan(min(positive_positions) - 0.5, max(positive_positions) + 0.5, 
                         alpha=0.2, color='skyblue')
        
        # Set title and labels
        axs[i].set_title(f'Chapter {chapter} Fever Chart', fontsize=12)
        axs[i].set_xlabel('Step', fontsize=10)
        
        # Set x-ticks
        axs[i].set_xticks(steps)
        axs[i].set_xticklabels([f'{s}' for s in steps])
        
        # Set proper y-axis limits
        axs[i].set_ylim(-0.5, len(unique_scores)-0.5)
        
        # Add grid
        axs[i].grid(True, alpha=0.3)
    
    # Add a legend at the bottom of the figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Moving Toward Positive'),
        Line2D([0], [0], color='red', lw=2, label='Moving Toward Negative'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Emotional States'),
        plt.Rectangle((0,0), 1, 1, alpha=0.2, color='darkred', label='Highly Negative'),
        plt.Rectangle((0,0), 1, 1, alpha=0.2, color='lightgray', label='Mixed/Neutral'),
        plt.Rectangle((0,0), 1, 1, alpha=0.2, color='skyblue', label='Positive')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    # Add a title for the entire figure
    plt.suptitle('Fever Chart Within Key Chapters of "The Picture of Dorian Gray"', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust the rect parameter to make room for suptitle and legend
    
    plt.savefig('actualFeverChart/all_chapter_progressions_updated.png', dpi=300)
    plt.close()

# Create the combined figure
create_all_chapter_progressions()