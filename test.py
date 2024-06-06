from pca import PCA_1
import numpy as np
from scipy.integrate import solve_ivp


# The function of the flow field
def vector_field(t, ys, vector=False):
    x, y = ys
    dx_dt = 2*x + y**2
    dy_dt = x**2 - y

    if not vector and (abs(x) > 5 or abs(y) > 5):
        return [0, 0]

    return [dx_dt, dy_dt]


x = np.linspace(-3, 5, 10)
y = np.linspace(-2, 5, 10)
X, Y = np.meshgrid(x, y)

t = np.linspace(0, 10, 1000)
xs = np.linspace(-0.95, -0.85, 50)
ys = np.repeat(-2.0, 50)

y0s = [[x, y] for x, y in zip(xs, ys)]

solutions = [solve_ivp(vector_field, (0, 10), y0, method="RK45", t_eval=t) for y0 in y0s]
solutions = [solution.y for solution in solutions]

# Interpolate the lines to make them equal length
point_amount = 300
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


# PCA Test
reshaped_data = np.reshape(interpolated_solutions, (50, -1))

pca = PCA_1(reshaped_data)

projected_data = pca.reduce_dimension(2)
