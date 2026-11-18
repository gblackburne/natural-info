import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

coordinates = [(3,0),
               (1,1), (2,1), (4,1), (5,1),
               (3,2),
               (0.15,2.5), (1.5,2.5), (4.5,2.5), (5.85,2.5),
               (3,3),
               (1,4), (2,4), (4,4), (5,4),
               (3,5)]
labels = ["rtr", 
          "rta", "rtb", "xtr", "ytr", 
          "xtb", 
          "rts", "xta", "ytb", "str",
          "yta", 
          "xts", "yts", "sta", "stb",
          "sts"]
label_coord = dict(zip(labels, coordinates))

edges = {("sts", "xts"), ("sts", "xts"), ("sts", "yts"), ("sts", "sta"), ("sts", "stb"),
         ("yta", "yts"), ("yta", "sta"), ("yta", "rta"), ("yta", "ytr"),
         ("xtb", "xts"), ("xtb", "stb"), ("xtb", "rtb"), ("xtb", "xtr"),
         ("xts", "sts"), ("xts", "rts"), ("xts", "xta"), ("xts", "xtb"),
         ("yts", "sts"), ("yts", "rts"), ("yts", "ytb"), ("yts", "yta"),
         ("sta", "sts"), ("sta", "xta"), ("sta", "yta"), ("sta", "str"),
         ("stb", "sts"), ("stb", "rtb"), ("stb", "ytb"), ("stb", "str"),
         ("rts", "xts"), ("rts", "yts"), ("rts", "rta"), ("rts", "rtb"),
         ("xta", "xts"), ("xta", "sta"), ("xta", "rta"), ("xta", "xtr"),
         ("ytb", "yts"), ("ytb", "stb"), ("ytb", "rtb"), ("ytb", "ytr"),
         ("str", "sta"), ("str", "stb"), ("str", "xtr"), ("str", "ytr"),
         ("rta", "rts"), ("rta", "xta"), ("rta", "yta"), ("rta", "rtr"),
         ("rtb", "rts"), ("rtb", "xtb"), ("rtb", "ytb"), ("rtb", "rtr"),
         ("xtr", "xta"), ("xtr", "xtb"), ("xtr", "str"), ("xtr", "rtr"),
         ("ytr", "yta"), ("ytr", "ytb"), ("ytr", "str"), ("ytr", "rtr"),
         ("rtr", "rta"), ("rtr", "rtb"), ("rtr", "xtr"), ("rtr", "ytr")}

def draw_lattice(label_coord=label_coord, edges=edges, dlab=True):
    """
    Draw PhiID lattice.

    Parameters
    ----------
    label_coord : dict
        Dictionary mapping atom labels to their coordinates.
    edges : set of tuples
        Set of edges representing connections between atoms.
    dlab : bool, optional
        Whether to display labels on the atoms. Default is True.
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    for edge in edges:
        origin, destination = edge
        x1, y1 = label_coord[origin]
        x2, y2 = label_coord[destination]
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1, zorder=1)
    for atom, (x, y) in label_coord.items():
        ax.scatter(x, y, s=70, color='white', edgecolor='black', linewidth=1)
        if dlab:
            label = atom.replace('x', r'$U_{1}$').replace('y', r'$U_{2}$')
            label = label.replace('a', r'$U_{1}$').replace('b', r'$U_{2}$')
            label = label.replace('r', '$R$').replace('s', '$S$')
            label = label.replace('t', '$\\rightarrow$')
            ax.text(x, y, label, ha='center', va='center', fontsize=12)
    ax.axis('off')
    return fig, ax


def draw_lattice_results(phiid, 
                         label_coord=label_coord, edges=edges,
                         colors='magma_r',
                         vmin=None, vmax=None,
                         ax=None, fs=15, ns=250, lw=1,
                         cbar_label=r'$I_{\partial}^{\alpha\rightarrow\beta}$',
                         addcstart=False,
                         cbar_kws={'pad':0.05, 'fraction':0.02, 'shrink':0.4, 'aspect':12.5},
                         set_ticks=None,
                         phiid_stars=None):
    """
    Draw PhiID lattice with results.

    Parameters
    ----------
    phiid : dict
        Dictionary mapping atom labels to their PhiID values.
    label_coord : dict
        Dictionary mapping atom labels to their coordinates.
    edges : set of tuples
        Set of edges representing connections between atoms.
    colors : str, optional
        Colormap to use.
    min : float, optional
        Minimum value for color scaling.
    max : float, optional
        Maximum value for color scaling.
    cbar_label : str, optional
        Label for the colorbar. 
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    fs : int, optional
        Font size for labels. Default is 15.
    ns : int, optional
        Node size. Default is 250.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.3, 5))
    colourmap = plt.get_cmap(colors)
    if addcstart is not False:
        if isinstance(colourmap, mpl.colors.ListedColormap):
            colours = colourmap.colors
        else:
            colours = colourmap(np.linspace(0, 1, colourmap.N))
        colours[0] = mpl.colors.to_rgba(addcstart)
        colourmap = mpl.colors.ListedColormap(colours)

    cmap = colourmap(np.linspace(0, 1, 1000))
    vmin = np.min(list(phiid.values())) if vmin is None else vmin
    vmax = np.max(list(phiid.values())) if vmax is None else vmax

    for edge in edges:
        origin, destination = edge
        x1, y1 = label_coord[origin]
        x2, y2 = label_coord[destination]
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=lw, zorder=1)

    for atom, (x, y) in label_coord.items():
        t = phiid[atom]
        colour = cmap[int((t-vmin)/(vmax-vmin) * (len(cmap) - 1))]
        ax.scatter(x, y, s=ns, color=colour, edgecolor='k', linewidth=lw, zorder=2)
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=colourmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', 
                        **cbar_kws)
    cbar.set_label(cbar_label, rotation=90, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    if set_ticks is not None:
        cbar.set_ticks(set_ticks)
    else:
        cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.draw() 
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_horizontalalignment('center')
    offset_text.set_position((0.5, 2))

    if phiid_stars is not None:
        for atom, star in phiid_stars.items():
            x, y = label_coord[atom]
            t = phiid[atom]
            colour = cmap[int((t-vmin)/(vmax-vmin) * (len(cmap) - 1))]
            brightness = np.sqrt(0.299*colour[0]**2 + 0.587*colour[1]**2 + 0.114*colour[2]**2)
            if brightness < 0.5:
                ax.text(x+.0275, y, star, 
                    ha='center', va='center', 
                    fontsize=fs*1.5, color='white', zorder=3)
            else:
                ax.text(x+.0275, y, star, 
                        ha='center', va='center', 
                        fontsize=fs*1.5, color='black', zorder=3)
    return ax
