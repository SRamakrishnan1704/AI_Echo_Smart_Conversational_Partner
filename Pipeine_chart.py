import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = r"D:\Ramakrishnan S\Guvi\Visual studio\My Project foler\Smartest_Conversational_Partner"

fig, ax = plt.subplots(figsize=(14, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

def draw_box(ax, x, y, w, h, label, sublabel='', color='#1C2A3A', border='#378ADD', text_color='#60A5FA'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color)
        ax.text(x, y - 0.2, sublabel, ha='center', va='center',
                fontsize=8, color='#8B949E')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color='#378ADD'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax.text(5, 19.3, 'Smartest Conversational Partner', ha='center',
        fontsize=16, fontweight='bold', color='#A78BFA')
ax.text(5, 18.9, 'End-to-End ML Pipeline', ha='center',
        fontsize=11, color='#8B949E')

draw_box(ax, 5, 18.2, 3.5, 0.6, 'Data Loading', 'loader.py',
         color='#161B22', border='#5F5E5A', text_color='#D3D1C7')
draw_arrow(ax, 5, 17.9, 5, 17.4)

draw_box(ax, 5, 17.1, 3.5, 0.6, 'Data Cleaning', 'Data_Cleaning.py',
         color='#0D2B22', border='#0F6E56', text_color='#5DCAA5')
draw_arrow(ax, 5, 16.8, 5, 16.3)

draw_box(ax, 5, 16.0, 3.5, 0.6, 'Exploratory Data Analysis', 'eda.py',
         color='#0D2B22', border='#0F6E56', text_color='#5DCAA5')
draw_arrow(ax, 5, 15.7, 5, 15.2)

draw_box(ax, 5, 14.9, 3.8, 0.6, 'Feature Extraction', 'Featureextraction_1.py',
         color='#1E1B4B', border='#534AB7', text_color='#AFA9EC')
draw_arrow(ax, 3.3, 14.6, 2.5, 13.9, color='#534AB7')
draw_arrow(ax, 6.7, 14.6, 7.5, 13.9, color='#534AB7')

draw_box(ax, 2.2, 13.5, 2.8, 0.7, 'TF-IDF Vectorizer', '5,000 features · bigrams',
         color='#1E1B4B', border='#534AB7', text_color='#AFA9EC')
draw_box(ax, 7.8, 13.5, 2.8, 0.7, 'BERT Embeddings', '768-dim · CLS token',
         color='#1E1B4B', border='#534AB7', text_color='#AFA9EC')
draw_arrow(ax, 2.8, 13.15, 3.8, 12.55, color='#534AB7')
draw_arrow(ax, 7.2, 13.15, 6.2, 12.55, color='#534AB7')

draw_box(ax, 5, 12.2, 3.8, 0.6, 'Model Training', 'Model_train.py',
         color='#0C1A2E', border='#185FA5', text_color='#85B7EB')
draw_arrow(ax, 3.2, 11.9, 1.8, 11.3, color='#185FA5')
draw_arrow(ax, 4.1, 11.9, 3.5, 11.3, color='#185FA5')
draw_arrow(ax, 5.0, 11.9, 5.0, 11.3, color='#185FA5')
draw_arrow(ax, 5.9, 11.9, 6.5, 11.3, color='#185FA5')
draw_arrow(ax, 6.8, 11.9, 8.2, 11.3, color='#185FA5')

models = [
    (1.5, 'LR + TF-IDF', 'Best'),
    (3.2, 'LR + BERT',   ''),
    (5.0, 'RF + TF-IDF', ''),
    (6.8, 'RF + BERT',   ''),
    (8.5, 'LSTM + BERT', ''),
]
for mx, mlabel, msub in models:
    bc = '#0D2B1A' if msub else '#0C1A2E'
    ec = '#059669' if msub else '#185FA5'
    tc = '#34D399' if msub else '#85B7EB'
    draw_box(ax, mx, 10.9, 1.55, 0.65, mlabel, msub,
             color=bc, border=ec, text_color=tc)

draw_arrow(ax, 1.5, 10.58, 3.5, 10.0, color='#059669')
draw_arrow(ax, 3.2, 10.58, 4.1, 10.0, color='#185FA5')
draw_arrow(ax, 5.0, 10.58, 5.0, 10.0, color='#185FA5')
draw_arrow(ax, 6.8, 10.58, 5.9, 10.0, color='#185FA5')
draw_arrow(ax, 8.5, 10.58, 6.5, 10.0, color='#185FA5')

draw_box(ax, 5, 9.65, 4.0, 0.65, 'Best Model Selected', 'LR + TF-IDF · fastest · lightest',
         color='#0D2B1A', border='#059669', text_color='#34D399')
draw_arrow(ax, 5, 9.33, 5, 8.78)

draw_box(ax, 5, 8.45, 4.0, 0.65, 'Prediction Script', 'Step3_Prediction.py + keyword fallback',
         color='#0D2B1A', border='#059669', text_color='#34D399')
draw_arrow(ax, 5, 8.13, 5, 7.58)

draw_box(ax, 5, 7.25, 4.2, 0.65, 'Streamlit Dashboard', 'app.py · 6 pages · 20+ charts',
         color='#2A1A00', border='#D97706', text_color='#FCD34D')
draw_arrow(ax, 3.1, 6.93, 2.0, 6.38, color='#D97706')
draw_arrow(ax, 5.0, 6.93, 5.0, 6.38, color='#D97706')
draw_arrow(ax, 6.9, 6.93, 8.0, 6.38, color='#D97706')

draw_box(ax, 1.8, 6.05, 2.8, 0.6, 'Overview + EDA', '',
         color='#2A1A00', border='#D97706', text_color='#FCD34D')
draw_box(ax, 5.0, 6.05, 2.8, 0.6, 'Predict Sentiment', '',
         color='#2A1A00', border='#D97706', text_color='#FCD34D')
draw_box(ax, 8.2, 6.05, 2.8, 0.6, 'Sentiment Q&A x10', '',
         color='#2A1A00', border='#D97706', text_color='#FCD34D')

legend_items = [
    (mpatches.Patch(color='#5F5E5A'), 'Data pipeline'),
    (mpatches.Patch(color='#0F6E56'), 'Cleaning & EDA'),
    (mpatches.Patch(color='#534AB7'), 'Feature extraction'),
    (mpatches.Patch(color='#185FA5'), 'Model training'),
    (mpatches.Patch(color='#059669'), 'Best model & prediction'),
    (mpatches.Patch(color='#D97706'), 'Dashboard'),
]
handles, labels = zip(*legend_items)
ax.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
          facecolor='#161B22', edgecolor='#30363D',
          labelcolor='#8B949E', bbox_to_anchor=(0.5, 0.01))

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'SCP_Pipeline.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='#0D1117', edgecolor='none')
plt.show()
print(f"Chart saved to {output_path}")