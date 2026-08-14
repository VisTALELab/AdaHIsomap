
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

    
def plot_original_data_in_2d(X, links, Landmark, color, title, output_dir, base_point = None, plot_title=None, show_skeleton="off"):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if plot_title is not None:
        ax.set_title(
            plot_title,
            fontsize=16,
            fontweight="bold",
            pad=20)

    ax.scatter(X[:, 0], X[:, 1],s=20, c=color,
               cmap=plt.get_cmap('coolwarm'))
    
    if  show_skeleton == "on":

        if base_point is not None:
            ax.scatter(base_point[0], base_point[1], c='red', s=100,  marker=(5, 1), label='Base Point')
            
        ax.scatter(Landmark[:, 0], Landmark[:, 1],
                   c='black', s=100,  marker=(5, 1), label='Landmarks(Centroid)')
        
        for i in range(0, len(links)):
            ax.plot([Landmark[links[i][0]][0], Landmark[links[i][1]][0]], [Landmark[links[i][0]][1],
                                                                       Landmark[links[i][1]][1]], color='black')

    plt.axis('equal')    
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    filename = os.path.join(output_dir, f"{title}_plot_original_data.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()



def plot_original_data_in_3d(X, links, Landmark, color, title, output_dir, base_point= None, plot_title=None, show_skeleton="off"):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if plot_title is not None:
        ax.set_title(
        plot_title,
        fontsize=16,
        fontweight="bold",
        pad=20)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=70, c=color, cmap=plt.get_cmap('coolwarm'), alpha=0.08, zorder=3)
        
    if  show_skeleton == "on":

        if base_point is not None:
            ax.scatter(base_point[0], base_point[1], base_point[2],
                       cmap=plt.cm.Spectral, c='red', s=140,  marker=(5, 1), label='Base Point', zorder=1)
            
        ax.scatter(Landmark[:, 0], Landmark[:, 1], Landmark[:, 2],
                   c='black', s=110,  marker=(5, 1), label='Landmarks',zorder=1)
        
        for i in range(0, len(links)):
            ax.plot([Landmark[links[i][0]][0], Landmark[links[i][1]][0]], [Landmark[links[i][0]][1],
                                                                       Landmark[links[i][1]][1]], [Landmark[links[i][0]][2], Landmark[links[i][1]][2]], color='black',zorder=1,linewidth=2)

    plt.axis('equal')    
    set_axes_equal(ax)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    filename = os.path.join(output_dir, f"{title}_plot_original_data.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()



def plot_projection(Y, proj_landmark, projected_stochastic_anchor, links, color, title, output_dir, BP_id, plot_title=None, show_skeleton="off"):
    
    base_point = None

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if plot_title is not None:
        ax.set_title(
        plot_title,
        fontsize=16,
        fontweight="bold",
        pad=20)

    ax.scatter(Y[:, 0].real, Y[:, 1].real, s=60, c=color,
               cmap=plt.get_cmap('coolwarm'))
    
    if  show_skeleton == "on":

        if BP_id is not None:
            base_point = Y[BP_id]
            ax.scatter(base_point[0], base_point[1],cmap=plt.cm.Spectral, c='red', s=200,  marker=(5, 1), label='Base Point')
        
        ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=120,  marker=(
        5, 1), label='Landmarks')

        ax.scatter(projected_stochastic_anchor[:, 0], projected_stochastic_anchor[:, 1], c='darkorange', s=100,  marker=(
        5, 1), edgecolors='black', linewidths=1.2, label='Stochastic Anchor Points')

        link_num = len(links)
        for i in range(link_num):
            ax.plot([proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [
                proj_landmark[links[i][0]][1], proj_landmark[links[i][1]][1]], color='black',linewidth=2.5)

    ax.legend(loc='best', fontsize=9, frameon=True, facecolor='white', edgecolor='black')
    plt.axis("off")  
    filename = os.path.join(output_dir, f"{title}_projection.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()



def plot_projection_by_index(Y, proj_landmark, projected_stochastic_anchor, links, title, output_dir, plot_title=None, show_skeleton="off"):
    

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if plot_title is not None:
        ax.set_title(
        plot_title,
        fontsize=16,
        fontweight="bold",
        pad=20)

    time_values = np.arange(len(Y))

    ax.scatter(Y[:, 0].real, Y[:, 1].real, s=50, c=time_values,
                          cmap=plt.get_cmap('coolwarm'),label = "Data points (colored by original index)",)
    

    if  show_skeleton == "on":
        
        ax.scatter(proj_landmark[:, 0], proj_landmark[:, 1], c='black', s=80,  marker=(
        5, 1), label='Landmarks')

        ax.scatter(projected_stochastic_anchor[:, 0], projected_stochastic_anchor[:, 1], c='darkorange', s=80,  marker=(
        5, 1), edgecolors='black', linewidths=1.2, label='Stochastic Anchor Points')

        link_num = len(links)
        for i in range(link_num):
            ax.plot([proj_landmark[links[i][0]][0], proj_landmark[links[i][1]][0]], [
                proj_landmark[links[i][0]][1], proj_landmark[links[i][1]][1]], color='black')


    ax.legend(loc='best', fontsize=9, frameon=True, facecolor='white', edgecolor='black')
    plt.axis("off")  
    filename = os.path.join(output_dir, f"{title}_projection_by_index.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()



def plot_projection_graph(Y, color, node_labels, graph, title, output_dir, show_skeleton="off"):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.set_title(
    "AdaMapper Skeleton + AdaHIsomap Projection",
    fontsize=16,
    fontweight="bold",
    pad=20)

    ax.scatter(Y[:, 0].real, Y[:, 1].real, s=520, c=color, edgecolors="black", linewidths=0.6, zorder=3)

    if show_skeleton == "on":
        for node_u, node_v in graph.edges():

            if (node_u in node_labels and node_v in node_labels):

                index_u = node_labels[node_u]
                index_v = node_labels[node_v]

                ax.plot(
                    [Y[index_u, 0], Y[index_v, 0]], [Y[index_u, 1], Y[index_v, 1]], color="black", alpha=0.5, linewidth=5, zorder=1)

    plt.axis("off")
    filename = os.path.join(output_dir, f"{title}_projection_graph.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()


def set_axes_equal(ax):
    """
    Set equal scaling for 3D plot.
    Ensures that the 3D axes are scaled equally for a cubic box.
    """
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    centers = np.mean(extents, axis=1)
    max_range = np.max(np.abs(extents[:, 1] - extents[:, 0])) / 2

    for center, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(center - max_range, center + max_range)