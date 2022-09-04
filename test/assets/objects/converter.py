import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def main():
  filename = "Shape1"
  mesh = o3d.io.read_triangle_mesh(filename+".stl")
  pcd = mesh.sample_points_uniformly(number_of_points=500)
  o3d.io.write_point_cloud(filename+".ply", pcd)

  # In bo.py
  new_pcd = o3d.io.read_point_cloud(filename+".ply")
  points = np.asarray(new_pcd.points)
  points = points/np.max(points,0)*0.1

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points[:,0], points[:,1], points[:,2])
  plt.show()


if __name__ == '__main__':
  main()