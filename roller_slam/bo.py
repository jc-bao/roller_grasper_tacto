import numpy as np
from scipy.spatial.transform import Rotation as R


from roller_slam.utils import load_data, get_hole_shape, get_section_points, GaussianProcess, Plotter, cartisian_to_spherical


def get_projection_shape(gp_shape, orn):
  projection_shape = None
  return projection_shape

def eval_projection(projection_shape, hole_shape):
  return False

def eval_shape(gp_shape, hole_shape):
  # enumerate all possible orientations
  all_orientations = []
  errs = []
  for orn in all_orientations():
    # get the projection shape
    projection_shape = get_projection_shape(gp_shape, orn)
    # check if the projection shape is a hole
    errs.append(eval_projection(projection_shape, hole_shape))
  return errs

def eval_section(gp_shape, insert_orn, hole_shape):
  # enumerate all possible section displacements
  all_displacements = []
  errs = []
  for disp in all_displacements:
    # get slice of gp_shape
    gp_shape_slice = None
    # get the new shape
    projection_shape = get_projection_shape(gp_shape, insert_orn)
    # check if the new shape is a hole
    errs.append(eval_shape(projection_shape, hole_shape))
  return errs

def conduct_bo(errs):
  return np.max(errs.std())

def main():
  # num_plts = 3
  # fig = plt.figure(figsize=(5, 5*num_plts))
  plotter = Plotter(num_figs=3)
  # load the data
  points_normed = load_data('../test/assets/bottle.ply')
  points_spherical = cartisian_to_spherical(points_normed)
  plotter.plot_sphere_coordinate(points_spherical, 'bottle point clouds')
  # ax = fig.add_subplot(num_plts, 1, 1, projection='3d')
  # ax.scatter(points_normed[:, 0], points_normed[:, 1], points_normed[:, 2])
  # ax.set_xlabel('x')
  # ax.set_ylabel('y')
  # ax.set_zlabel('z')
  # ax.set_box_aspect((np.ptp(points_normed[:, 0]), np.ptp(points_normed[:, 1]), np.ptp(points_normed[:,2])))

  # get the hole shape
  hole_euler = np.array([0, 0, 0])
  hole_shape = get_hole_shape(points_normed, R.from_euler('xyz', hole_euler))
  plotter.plot_2d(hole_shape, 'hole shape')
  # ax = fig.add_subplot(num_plts, 1, 2)
  # ax.plot(hole_shape[:, 0], hole_shape[:, 1])
  # ax.set_xlabel('x')
  # ax.set_ylabel('y')
  # ax.set_title(f'hole shape, orn={hole_euler}')

  # explore the object
  explore_section_pos = np.array([0,0,0]) # explore the center section first
  section_points = get_section_points(points_normed, explore_section_pos)
  plotter.plot_3d(section_points, f'section points, orn={explore_section_pos}')
  # ax = fig.add_subplot(num_plts, 1, 3, projection='3d')
  # ax.scatter(section_points[:, 0], section_points[:, 1], section_points[:, 2])
  # ax.set_xlabel('x')
  # ax.set_ylabel('y')
  # ax.set_zlabel('z')
  # ax.set_title(f'section points, orn={explore_section_pos}')
  # ax.set_box_aspect((np.ptp(section_points[:, 0]), np.ptp(section_points[:, 1]), np.ptp(section_points[:,2])))

  # fit a probabilistic model to the section points
  # gp_shape = GaussianProcess(section_points[:,:2], section_points[:,2])

  # evaluate the shape
  # Question: use joint distribution v.s. use marginal distribution to evaluate model?
  # Answer: if we care about the whole object, then we need to use the joint distribution. If we only care about one point, then marginal distribution is fine. In this case, we use the joint distribution to evaluate the whole objects. 
  # Pipeline: 1. get the joint distribution of the object; 2. project this distribution into a orientation; 3. get the hull distribution of the projected object; 4. evaluate the shape of the hull distribution compared with the hole shape.

  plotter.save('../test/results/bottle_BO.png')


if __name__ == '__main__':
  main()
