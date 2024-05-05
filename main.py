import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


# Define the vector field function F(x, y) = (-y, x)
def vector_field(x, y):
    return (2*x+y, x-y)


# Define the grid of points
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)

# Evaluate the vector field at each point in the grid
U, V = vector_field(X, Y)

# Plot the flow field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, scale=20)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flow Field: F(x, y) = (-y, x)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.show()
