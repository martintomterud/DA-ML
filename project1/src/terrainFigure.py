from imageio import imread
import matplotlib.pyplot as plt
import matplotlib
#matplotlib updates
matplotlib.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
import matplotlib.transforms as mtransforms
import dataFunctions
import os

"""
This python file produces the figure in the report
that shows how we have selected our terrain data
"""

#path for storing figures
cd = os.path.abspath('..')
datapath = cd + '/data/'
dataname = 'SRTM_data_Norway_1.tif'

size = 1000 # M x M size of terrain region
N = 100
q = int(size/N)
terrain = dataFunctions.importTerrain(datapath + dataname)
terrainRegion = dataFunctions.terrainGridRegion(terrain, size)
scaledTerrainRegion = dataFunctions.scaleTerrain(terrainRegion)

"""
Setup taken from matplotlib doc
"""

fig, axs = plt.subplot_mosaic([['a)', 'b)'], ['a)', 'c)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    ax.set_xticks([])
    ax.set_yticks([])

axs['a)'].imshow(terrain, cmap = 'winter')
axs['b)'].imshow(terrainRegion, cmap = 'winter')
axs['c)'].imshow(terrainRegion[::q, ::q], cmap = 'winter')
#plt.savefig(cd + '/figures/' + 'terrain_select.pdf', dpi = 800, bbox_inches = 'tight')
plt.show()