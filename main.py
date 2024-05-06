import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("TkAgg")


# The function of the flow field
def vector_field(t, y):
    x, y = y
    dx_dt = 2*x + y**2
    dy_dt = x**2 - y

    return [dx_dt, dy_dt]

# Plot settings
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flow Field: F(x, y) = (2*x + y**2, x**2 - y)')
plt.xlim(-3, 5)
plt.ylim(-2, 5)
plt.grid(True)

# Plot arrows showing the direction of the flow field
x = np.linspace(-3, 5, 10)
y = np.linspace(-2, 5, 10)
X, Y = np.meshgrid(x, y)
U, V = vector_field(0, (X, Y))
plt.quiver(X, Y, U, V, scale=200)


# Integrate over the flow field using the Runge-Kutta method
t = np.linspace(0, 10, 1000)
xs = np.linspace(-0.9, -0.85, 20)
ys = np.repeat(-2.0, 20)

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
    plt.plot(solution[0], solution[1])


plt.show()
