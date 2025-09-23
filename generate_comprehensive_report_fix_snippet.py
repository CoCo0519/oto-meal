
# ===== Drop-in snippet for complete_training_pipeline.py =====
# Find the bar plot where class F1 is drawn, replace that block with:

colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
n = len(class_names)
if n == 0:
    # No classes, skip plotting to avoid errors
    axes[1, 1].text(0.5, 0.5, 'No classes to plot', ha='center', va='center')
else:
    if len(colors) < n:
        times = (n + len(colors) - 1) // len(colors)
        colors = (colors * times)[:n]
    else:
        colors = colors[:n]
    axes[1, 1].bar(class_names, class_f1s, color=colors)
    axes[1, 1].set_ylabel('F1-score')
    axes[1, 1].set_title('Per-class F1')
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, axis='y', linestyle=':', alpha=0.5)

# (If there are other bar/line plots that pass a fixed-length color list,
#  apply the same pattern: ensure len(color_list) matches len(data).)
# =============================================================
