import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA


matplotlib.use("TkAgg")


# The function of the flow field
def vector_field(t, y):
    x, y = y
    dx_dt = 2*x + y**2
    dy_dt = x**2 - y

    return [dx_dt, dy_dt]

x = np.linspace(-3, 5, 10)
y = np.linspace(-2, 5, 10)
X, Y = np.meshgrid(x, y)

U, V = vector_field(0, (X, Y))

figure, axis = plt.subplots(1, 2)

axis[0].quiver(X, Y, U, V, scale=200)
axis[0].set_xlabel('x')
axis[0].set_ylabel('y')
axis[0].set_title('Flow Field: F(x, y) = (2*x + y**2, x**2 - y)')
axis[0].set_xlim(-3, 5)
axis[0].set_ylim(-2, 5)
axis[0].grid(True)

t = np.linspace(0, 5, 1000)
xs = np.linspace(-0.95, -0.85, 50)
ys = np.repeat(-2.0, 50)

y0s = [[x, y] for x, y in zip(xs, ys)]

solutions = [solve_ivp(vector_field, (0, 10), y0, method="RK45", t_eval=t) for y0 in y0s]
solutions = [solution.y for solution in solutions]

# Downsample the solutions in order to get lines of the same dimension
min_len = min(len(solution[0]) for solution in solutions)
downsampled_solutions = []
for solution in solutions:
    downsampled_solution = [[], []]
    solution_length = len(solution[0])
    step = solution_length / min_len
    i = 0

    while round(i) < solution_length:
        downsampled_solution[0].append(solution[0][round(i)])
        downsampled_solution[1].append(solution[1][round(i)])
        i += step

    downsampled_solutions.append(downsampled_solution)

# Plot the downsampled solutions
for solution in downsampled_solutions:
    axis[0].plot(solution[0], solution[1])


# PCA Test
reshaped_data = np.reshape(downsampled_solutions, (50, -1))

pca = PCA(n_components=2)
projected_data = pca.fit_transform(reshaped_data)

axis[1].scatter(projected_data[:, 0], projected_data[:, 1])
axis[1].set_xlabel('Principal Component 1')
axis[1].set_ylabel('Principal Component 2')
axis[1].set_title('Scatter Plot of Projected Lines')

plt.show()
