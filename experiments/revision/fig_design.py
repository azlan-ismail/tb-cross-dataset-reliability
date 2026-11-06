import sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = Path(sys.argv[1])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'text.color': '#222222'})
INK, MUTED = '#222222', '#555555'
GREY_F, GREY_E = '#f1f2f4', '#8a8f98'
BLUE_F, BLUE_E = '#e6f0f8', '#0072B2'
ORNG_F, ORNG_E = '#fbece3', '#D55E00'
GRN_F, GRN_E = '#e5f4ef', '#009E73'
fig = plt.figure(figsize=(7.6, 6.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 91); ax.axis('off')

def box(x, y, w, h, title, body='', fc=GREY_F, ec=GREY_E, ls='-', tsize=9.6, bsize=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc=fc, ec=ec, lw=1.1, ls=ls, zorder=2))
    cx = x + w / 2
    if body:
        ax.text(cx, y + h - 2.0, title, ha='center', va='top', fontsize=tsize, fontweight='bold', color=INK, zorder=3)
        ax.text(cx, y + h - 5.4, body, ha='center', va='top', fontsize=bsize, color=INK, linespacing=1.35, zorder=3)
    else:
        ax.text(cx, y + h / 2, title, ha='center', va='center', fontsize=tsize, fontweight='bold', color=INK, zorder=3)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, ha='center', rad=0.0, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11, lw=1.2,
                                 color='#444444', zorder=z, connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha=ha, va='center', fontsize=8.2,
                color=MUTED, style='italic', zorder=3,
                bbox=dict(fc='white', ec='none', pad=1.2))

def header(x, y, text):
    ax.text(x, y, text, ha='left', va='bottom', fontsize=10, fontweight='bold', color=MUTED)

# column headers
header(1.5, 86.5, '1  Development (Shenzhen)')
header(35.5, 86.5, '2  Model and decision rule')
header(69.5, 86.5, '3  External evaluation')

# column 1
box(4, 72, 26.5, 13, 'Shenzhen', '662 images\n336 TB, 326 normal')
box(4, 52, 26.5, 13, 'Training split', '529 images\n267 TB, 262 normal')
box(4, 32, 26.5, 13, 'Validation split', '133 images\n69 TB, 64 normal')
arrow(17.25, 72, 17.25, 65.2)
ax.plot([4, 1.8, 1.8], [78.5, 78.5, 38.5], color='#444444', lw=1.2, zorder=1, solid_capstyle='butt')
arrow(1.8, 38.5, 4, 38.5)
ax.text(17.25, 48.5, 'stratified, group-aware\nsplit', ha='center', va='center', fontsize=8.2, color=MUTED, style='italic')

# column 2
box(35.5, 72, 29, 13, 'Frozen RAD-DINO encoder', '768-dimensional\nclass-token features')
box(35.5, 52, 29, 13, 'Classifiers', 'Linear: logistic regression\nNeural: 1 hidden layer,\nseeds 42, 43, 44')
box(35.5, 29.5, 29, 15.5, 'Threshold selection', 'highest threshold with\n$\\geq$90% validation sensitivity\n(+ 10,000 bootstrap\nre-selections)')
arrow(30.5, 58.5, 35.5, 58.5, 'fit', ly=1.9)
arrow(50, 72, 50, 65.2)
arrow(50, 52, 50, 45.2)
arrow(30.5, 38.5, 35.5, 38.5, 'select', ly=2.4)

# column 3
box(69.5, 72, 29, 13, 'Montgomery', '138 images\n58 TB, 80 normal', fc=BLUE_F, ec=BLUE_E)
box(69.5, 29.5, 29, 35.5, 'TBX11K validation', ' ', fc=ORNG_F, ec=ORNG_E)
ax.text(84, 59.0, '1,800 images', ha='center', va='top', fontsize=8.6)
arrow(76, 56.0, 76, 50.2, z=3)
ax.text(78, 53.0, '6 exact duplicates\nremoved', ha='left', va='center', fontsize=8.0, color=MUTED, style='italic')
ax.text(84, 49.2, '1,794 images', ha='center', va='top', fontsize=8.6, fontweight='bold')
for i, (lab, n) in enumerate([('TB', '200'), ('Healthy', '800'), ('Other illness', '794')]):
    yy = 45.3 - i * 4.6
    ax.add_patch(FancyBboxPatch((73, yy - 3.4), 22, 3.6, boxstyle='round,pad=0,rounding_size=0.8',
                                fc='white', ec=ORNG_E, lw=0.8, zorder=2))
    ax.text(74.5, yy - 1.6, lab, ha='left', va='center', fontsize=8.4, zorder=3)
    ax.text(93.5, yy - 1.6, n, ha='right', va='center', fontsize=8.4, fontweight='bold', zorder=3)
# transfer arrows
ax.plot([64.5, 67, 67], [37, 37, 78.5], color='#444444', lw=1.2, zorder=1)
arrow(67, 78.5, 69.5, 78.5)
arrow(67, 37, 69.5, 37)
ax.text(67, 58, 'applied unchanged', ha='center', va='center', fontsize=8.2, color=INK, fontweight='bold',
        rotation=90, bbox=dict(fc='white', ec='none', pad=1.0), zorder=3)

# analyses
header(1.5, 23.5, '4  Analyses on both external cohorts')
W = 23.125; G = 1.5
items = [('A  Operating point', 'AUROC, sensitivity\nand specificity at the\ntransferred threshold'),
         ('B  Threshold uncertainty', 'joint bootstrap interval;\nlocal reference threshold'),
         ('C  Negative spectrum', 'TBX11K healthy vs other\nillness; referral rate'),
         ('D  Calibration', 'Brier score, ECE,\ncalibration slope')]
for i, (t, b) in enumerate(items):
    box(1.5 + i * (W + G), 9.5, W, 13, t, b, fc=GRN_F, ec=GRN_E, tsize=8.6, bsize=8.2)
box(1.5, 1.2, 97.0, 6.3,
    'Secondary: RAD-DINO vs DINOv2 at 224 and 518 pixels, identical preprocessing and linear classifier',
    fc='white', ec=GREY_E, ls=(0, (4, 2.5)), tsize=8.6)
fig.savefig(OUT / 'Fig1.pdf'); fig.savefig(OUT / 'Fig1.png', dpi=200)
