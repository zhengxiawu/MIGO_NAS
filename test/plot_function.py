import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def function(X, Y):
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5. / np.pi
    r = 6.0
    s = 10.0
    t = 1. / (8. * np.pi)
    temp1 = (Y - b * X ** 2. + c * X - r)
    temp2 = np.cos(X)
    temp3 = a * temp1 ** 2
    temp4 = s * (1 - t) * temp2 + s
    y = temp3 + temp4
    return y


fig = plt.figure()
ax = Axes3D(fig)

# X, Y value
X = np.arange(-5, 10, 1)
Y = np.arange(0, 15, 1)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
Z = function(X, Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()
pass