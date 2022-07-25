import torch
import numpy as np
import os
import glob
import random
import matplotlib
import imageio
matplotlib.use('Agg')
import matplotlib.pyplot as plt
##############
### Files ####
###########-->


bond1_radius = {'H': 31, 'C': 76, 'N': 71, 'O': 66, 'F': 57} # covalnt bond in pm for each type of atom https://en.wikipedia.org/wiki/Covalent_radius
bond1_stdv = {'H': 5, 'C': 2, 'N': 2, 'O': 2, 'F': 3}

bond2_radius = {'H': -1000, 'C': 67, 'N': 60, 'O': 57, 'F': 59}
bond3_radius = {'H': -1000, 'C': 60, 'N': 54, 'O': 53, 'F': 53} # Not sure why oxygen has triple bond

atom_decoder = ['H', 'C', 'N', 'O', 'F']
atom_decoder_int = [1, 6, 7, 8, 9]

bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
          'C': {'H': 109, 'C': 154 , 'N': 147, 'O': 143, 'F': 135},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142}}

bonds2 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 134, 'N': 129, 'O': 120, 'F': -1000},
          'N': {'H': -1000, 'C': 129, 'N': 125, 'O': 121, 'F': -1000},
          'O': {'H': -1000, 'C': 120, 'N': 121, 'O': 121, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}

bonds3 = {'H': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000},
          'C': {'H': -1000, 'C': 120, 'N': 116, 'O': 113, 'F': -1000},
          'N': {'H': -1000, 'C': 116, 'N': 110, 'O': -1000, 'F': -1000},
          'O': {'H': -1000, 'C': 113, 'N': -1000, 'O': -1000, 'F': -1000},
          'F': {'H': -1000, 'C': -1000, 'N': -1000, 'O': -1000, 'F': -1000}}
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0


def save_xyz_file(
        path, one_hot, charges, positions, id_from=0, name='molecule'):
    try:
        os.makedirs(path)
    except OSError:
        pass
    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % one_hot.size(1))
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in range(one_hot.size(1)):
            atom = atoms[atom_i]
            atom = atom_decoder[atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()





def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files

#<----########
### Files ####
##############
def draw_sphere(ax, x, y, z, size, color):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v))
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0,
                    alpha=1.)


def plot_data3d(positions, atom_type, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False, bg='black'):

    black = (0, 0, 0)
    white = (1, 1, 1)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    #ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine
    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")
    #ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'])
    radius_dic = np.array([0.46, 0.77, 0.77, 0.77, 0.77])
    area_dic = 1500 * radius_dic ** 2
    #areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9, c=colors)#, linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            if get_bond_order(atom1, atom2, dist):
                if bg == 'black':
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], linewidth=(3-2)*2 * 2, c='#FFFFFF')
                else:
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                            linewidth=(3 - 2) * 2 * 2, c='#666666')

    axis_lim = 3.2
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 100 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.

        ax.imshow(im)

    plt.show()