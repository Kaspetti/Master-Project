import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

matplotlib.use("TkAgg")


def vector_field(y, t):
    x, y = y
    dx_dt = 2*x+y
    dy_dt = x-y

    return [dx_dt, dy_dt]


x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)

U, V = vector_field((X, Y), 0)

plt.quiver(X, Y, U, V, scale=20)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flow Field: F(x, y) = (2*x+y, x-y)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)

t = np.linspace(0, 10, 100)
xs = np.linspace(0.6, 0.65, 20)
ys = np.repeat(-2.0, 20)

y0s = [[x, y] for x, y in zip(xs, ys)]

solutions = [odeint(vector_field, y0, t) for y0 in y0s]

for solution in solutions:
    x_solution, y_solution = solution[:, 0], solution[:, 1]
    plt.plot(x_solution, y_solution)

plt.show()
