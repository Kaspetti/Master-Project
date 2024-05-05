import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("TkAgg")


def vector_field(t, y):
    x, y = y
    dx_dt = 2*x + y**2
    dy_dt = x**2 - y

    return [dx_dt, dy_dt]


x = np.linspace(-3, 5, 10)
y = np.linspace(-2, 5, 10)
X, Y = np.meshgrid(x, y)

U, V = vector_field(0, (X, Y))

plt.quiver(X, Y, U, V, scale=200)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flow Field: F(x, y) = (2*x + y**2, x**2 - y)')
plt.xlim(-3, 5)
plt.ylim(-2, 5)
plt.grid(True)

t = np.linspace(0, 10, 1000)
xs = np.linspace(-0.9, -0.85, 20)
ys = np.repeat(-2.0, 20)

y0s = [[x, y] for x, y in zip(xs, ys)]

solutions = [solve_ivp(vector_field, (0, 10), y0, method="RK45", t_eval=t) for y0 in y0s]

for solution in solutions:
    plt.plot(solution.y[0], solution.y[1])

plt.show()
