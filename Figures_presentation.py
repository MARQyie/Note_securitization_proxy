# --------------------------------------------
# Import Packages
# --------------------------------------------

# Data manipulation
# import pandas as pd
# import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=2.25)

# Set WD
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\01-Note_on_securitization')

# --------------------------------------------
# Set values
# --------------------------------------------

proxies = [14,5,2,2,2,1]
data = [5,1,4,2,7,8,1,1,1,4]

proxies_labels = ['Principle Balance Assets Sold and Securitized',
                  'Securities Issued',
                  'Total Assets ABCP Conduits',
                  'Guarantees to ABCP Conduits',
                  'Other',
                  'ABCPs Outstanding']

# --------------------------------------------
# Make plots
# --------------------------------------------

# Set values
bottom = 1
width = .001
colors = sns.color_palette('pastel')[0:6]

# Plot proxies
fig, ax  = plt.subplots(figsize = (6,12))
ax.set_frame_on(False)

for j, (height, label, color) in enumerate([*zip(proxies, proxies_labels,colors)]):
    bottom -= height / sum(proxies)
    bc = ax.bar(0,
                height / sum(proxies),
                width,
                bottom=bottom,
                color=color,
                label=label,
                alpha = 0.75)
    ax.bar_label(bc,
                 labels=[f"{height/ sum(proxies):.0%}"],
                 label_type='center')
    ax.text(width / 2,
            bottom + (height / sum(proxies)) / 2,
            label,
            wrap = True,
            va = 'center')

ax.axes.yaxis.set_visible(False)
ax.axes.xaxis.set_visible(False)
plt.tight_layout()

fig.savefig('Figures/Presentation_number_proxies_lit.png')