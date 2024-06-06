import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pca import PCA_1



# The function of the flow field
def vector_field(t, ys, vector=False):
    x, y = ys
    dx_dt = 2*x + y**2
    dy_dt = x**2 - y

    if not vector and (abs(x) > 5 or abs(y) > 5):
        return [0, 0]

    return [dx_dt, dy_dt]


def integrate_flow(point_amount):
    t = np.linspace(0, 10, 1000)
    xs = np.linspace(-0.95, -0.85, 50)
    ys = np.repeat(-2.0, 50)

    y0s = [[x, y] for x, y in zip(xs, ys)]

    solutions = [solve_ivp(vector_field, (0, 10), y0, method="RK45", t_eval=t) for y0 in y0s]
    solutions = [solution.y for solution in solutions]

    # Interpolate the lines to make them equal length
    interpolated_solutions = []
    for solution in solutions:
        x = solution[0]
        y = solution[1]

        indices = np.arange(len(x))
        new_indices = np.linspace(0, len(x) - 1, point_amount)

        interp_x = np.interp(new_indices, indices, x)
        interp_y = np.interp(new_indices, indices, y)

        interpolated_line = np.vstack((interp_x, interp_y))
        interpolated_solutions.append(interpolated_line)

    return interpolated_solutions


if __name__ == "__main__":
    matplotlib.use("TkAgg")

    x = np.linspace(-3, 5, 10)
    y = np.linspace(-2, 5, 10)
    X, Y = np.meshgrid(x, y)

    U, V = vector_field(0, (X, Y), vector=True)

    figure, axis = plt.subplots(1, 2)

    axis[0].quiver(X, Y, U, V, scale=200)
    axis[0].set_xlabel('x')
    axis[0].set_ylabel('y')
    axis[0].set_title('Flow Field: F(x, y) = (2*x + y**2, x**2 - y)')
    axis[0].set_xlim(-3, 5)
    axis[0].set_ylim(-2, 5)
    axis[0].grid(True)


    point_amount = 300
    interpolated_solutions = integrate_flow(point_amount)


# PCA Test
    reshaped_data = np.reshape(interpolated_solutions, (50, -1))

    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(reshaped_data)

# pca_1 = PCA_1(reshaped_data)
# n = 2
# projected_data_1 = pca_1.fit_transform(n)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(projected_data)

    cluster_colors = ["blue", "red"]
    center_colors = ["cyan", "orange"]

    cmap = mcolors.ListedColormap(cluster_colors)
    axis[1].scatter(projected_data[:, 0], projected_data[:, 1], c=kmeans.labels_, cmap=cmap)
    axis[1].set_xlabel('Principal Component 1')
    axis[1].set_ylabel('Principal Component 2')
    axis[1].set_title('Scatter Plot of Projected Lines')

# dings = pca_1.reconstruct(projected_data_1, 50)

    for i, solution in enumerate(interpolated_solutions):
        axis[0].plot(solution[0], solution[1], color=cluster_colors[kmeans.labels_[i]])

    axis[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c=center_colors, s=100)

    for i, center in enumerate(kmeans.cluster_centers_):
        # center_line = pca_1.reconstruct(center.reshape(1, n), n)
        # center_line = center_line.reshape(2, 300)
        # axis[0].plot(center_line[0], center_line[1], color=center_colors[i], linewidth=3)

        center_line = pca.inverse_transform(center)

        center_line = np.reshape(center_line, (2, point_amount))
        axis[0].plot(center_line[0], center_line[1], color=center_colors[i], linewidth=3)

    plt.show()
