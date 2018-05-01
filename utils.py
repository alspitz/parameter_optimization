import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def plot3(tss, vecs, labels):
  for i in range(3):
    pyplot.figure()

    for ts, vec, label in zip(tss, vecs, labels):
      pyplot.plot(ts, vec[:, i], label=label)

    pyplot.xlabel('Time (s)')
    pyplot.ylabel('xyz'[i] + ' (m)')
    pyplot.title('XYZ'[i] + ' Position')
    pyplot.legend()

  pyplot.show()

def plot_traj(pos):
  fig = pyplot.figure()
  ax = Axes3D(fig)
  ax.plot(pos[:, 0], pos[:, 1], zs=pos[:, 2])
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_zlabel('z (m)')
  pyplot.title("3D Trajectory")
  pyplot.show()

def scale_to_length(vec, length):
  return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(vec)), vec)

def scale_to_length_vecs(vec, length):
  interp = np.empty((length, vec.shape[1]))
  for i in range(vec.shape[1]):
    interp[:, i]  = scale_to_length(vec[:, i], length)
  return interp
