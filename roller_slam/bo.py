import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import torch, gpytorch
import open3d as o3d 
from tqdm import trange

from roller_slam.utils import load_data, get_hole_shape, get_section_points, GaussianProcess, Plotter, cartisian_to_spherical, spherical_to_cartisian, cartisian_to_polar, get_sphere_pcd, get_cylinder_pcd, get_cone_pcd, get_cube_pcd

def get_projection_shape(gp_shape, orn):
  """Project a 3d Guassian Distribution to a 2d Guassian Distribution
  

  Args:
      gp_shape (_type_): _description_
      orn (_type_): _description_

  Returns:
      _type_: _description_
  """
  projection_shape = None
  return projection_shape

def eval_object():
  pass

def eval_shape(sec_mu, sec_var):
  x_min = torch.linspace(-0.4, 0.4, 101).unsqueeze(-1)
  dist = torch.tensor(sec_mu)
  dist_std = torch.sqrt(torch.tensor(sec_var))
  x_normed = (x_min - dist) / dist_std # Tensor(n_x, n_theta)
  log_cdf = gpytorch.log_normal_cdf(x_normed)
  mask = log_cdf > -5
  cdf = torch.zeros_like(log_cdf)
  cdf[mask] = torch.exp(log_cdf[mask])
  cdf = torch.prod(1 - cdf, dim=1) # Tensor(n_x
  pdf = cdf[:-1] - cdf[1:]
  pdf = pdf / pdf.sum()
  mean = torch.sum(pdf * x_min[:-1,0])
  var = torch.sum(pdf * ((x_min[:-1,0] - mean)**2))
  return mean, var

def eval_section(y_mu, y_var, hole_shape_polar):
  assert y_mu.shape[0] == hole_shape_polar.shape[
      0], "The number of points in the hole shape and the Gaussian Process must be the same"
  x_min = torch.linspace(-0.4, 0.4, 101).unsqueeze(-1)
  dist = torch.tensor(hole_shape_polar[:,1] - y_mu)
  dist_std = torch.sqrt(torch.tensor(y_var))
  x_normed = (x_min - dist) / dist_std # Tensor(n_x, n_theta)
  log_cdf = gpytorch.log_normal_cdf(x_normed)
  mask = log_cdf > -5
  cdf = torch.zeros_like(log_cdf) # probability of x_min < X
  cdf[mask] = torch.exp(log_cdf[mask])
  cdf = torch.prod(1 - cdf, dim=1) # Tensor(n_x) p(x_min > X)
  pdf = cdf[:-1] - cdf[1:] 
  pdf = pdf / pdf.sum()
  mean = torch.sum(pdf * x_min[:-1,0])
  var = torch.sum(pdf * ((x_min[:-1,0] - mean)**2))
  return mean, var

def run_bo(init_explore_section = np.array([np.pi/2, np.pi/3, 0]), UCB_alpha = 500, if_plot = False, object_name='Shape1', silent=True, max_explore_time = 10):
  # object SLAM parameters
  n_phi = 20
  n_theta = n_phi * 2
  section_width = 0.025
  angle_step = torch.pi/n_phi
  hole_angle = np.array([0, 0])
  points_mean = 0
  points_std = 0.08
  # init_explore_section = np.array([np.pi, np.pi/3, 0]) # swap[2pi, pi, -0.06,-0.03,0.03,0.06]
  if UCB_alpha < 0:
    explore_policy = 'random' # swap 
  else:
    explore_policy = 'bo'
  # UCB_alpha = 100 # swap range [0, 100, 10000] 
  stop_bar = -0.03

  # torch
  torch.set_num_threads(2)

  plotter = Plotter(num_figs=np.array([max_explore_time+2, 7],dtype=np.int), if_plot=if_plot)
  # load the data
  new_pcd = o3d.io.read_point_cloud(f"../test/assets/objects/{object_name}.ply")
  points = np.asarray(new_pcd.points)
  points -= points.mean(axis=0)
  points = points/np.max(points,0)*0.1

  new_pcd_dense = o3d.io.read_point_cloud(f"../test/assets/objects/{object_name}_dense.ply")
  points_dense = np.asarray(new_pcd_dense.points)
  points_dense -= points_dense.mean(axis=0)
  points_dense = points_dense/np.max(points_dense,0)*0.1

  hole_rot = R.from_euler('zy', hole_angle)
  plotter.plot_3d([points_dense], 'object point clouds', true_aspect=True, plane_pose=np.append(hole_angle,0), alpha=[0.1])

  # get the hole shape
  hole_shape, hole_shape_polar = get_hole_shape(points_dense, hole_rot)
  if hole_shape_polar[-1,0] - hole_shape_polar[0,0] < 2*np.pi: 
    left_point = hole_shape_polar[-1].copy()
    left_point[0] -= 2 * np.pi
    right_point = hole_shape_polar[0].copy()
    right_point[0] += 2 * np.pi
    hole_shape_polar = np.concatenate([[left_point], hole_shape_polar, [right_point]], axis=0)
  # interpolate the hole shape to the the same resolution as the evaluation grid
  hole_shape_interp = interp1d(hole_shape_polar[:, 0], hole_shape_polar[:, 1], kind='cubic')
  hole_shape_polar= np.stack([np.linspace(-np.pi, np.pi, n_theta), hole_shape_interp(np.linspace(-np.pi, np.pi, n_theta))], axis=1)
  plotter.plot_2d([hole_shape], 'hole shape')
  # plotter.plot_polar(hole_shape_polar, 'hole shape polar')

  # evaluate the object
  obj_err_mu = np.zeros((n_theta, n_phi))
  pred_point = points
  pred_point_spherical = cartisian_to_spherical(pred_point)
  pred_var = np.zeros(pred_point.shape[0])
  all_ori = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, n_theta), np.linspace(-np.pi/2, np.pi/2, n_phi)), axis=-1).transpose(1,0,2)
  for i in range(n_theta):
    for j in range(n_phi):
      rot = R.from_euler('zy', -all_ori[i,j], degrees=False)
      pred_point_rot = (rot.apply(pred_point))
      pred_point_spherical_rot = cartisian_to_spherical(pred_point_rot)
      pred_point_rot = spherical_to_cartisian(pred_point_spherical_rot)
      # evaluate different sections
      sec_dist = 100
      for section_phi_idx, section_start in enumerate(np.arange(np.min(pred_point_rot[...,2]), np.max(pred_point_rot[...,2]), section_width)):
        section_end = section_start + section_width
        mask = (pred_point_rot[...,2] > section_start) & (pred_point_rot[...,2] < section_end)
        phi = pred_point_spherical_rot[mask, 1]
        theta = pred_point_spherical_rot[mask, 0]
        section_mu = pred_point_spherical_rot[mask, 2] * np.cos(phi)
        section_var = pred_var[mask] * (np.cos(phi)**2)
        hole_shape_polar= np.stack([theta, hole_shape_interp(theta)], axis=1)
        err_mu = np.min(hole_shape_polar[:,1] - section_mu)
        sec_dist = min(sec_dist, err_mu)
      obj_err_mu[i,j] = sec_dist
  data = np.concatenate([all_ori, obj_err_mu.reshape(n_theta, n_phi, 1)], axis=-1)
  true_object_err = obj_err_mu.copy()
  plotter.plot_heat(data, 'object evaluation result', axis_name=['theta', 'phi'])

  explore_section_pos = [init_explore_section] # explore the center section first
  min_dist_mu = []
  min_dist_var = []
  for explore_step in range(max_explore_time):
    # explore the object
    plotter.row = explore_step + 2
    section_points = get_section_points(points, explore_section_pos, section_width)
    plotter.plot_3d([section_points, points], f'section points, \n orn={explore_section_pos[-1]}', true_aspect=True, plane_pose=explore_section_pos[-1], alpha=[1,0.1])

    # fit a probabilistic model to the section points
    normed_section_points = (section_points - points_mean) / points_std
    normed_section_point_spherical = cartisian_to_spherical(normed_section_points)
    normed_section_point_tensor = torch.tensor(normed_section_point_spherical, dtype=torch.float32)
    try:
      gp_shape = GaussianProcess(normed_section_point_tensor[:,:2], normed_section_point_tensor[:,2], train_num=100, silent=silent)
    except Exception as e:
      print(e, 'gaussian model encounter an error')
      return -1, -1
    x_pred, y_pred_mu, y_pred_var = gp_shape.predict(step=angle_step)
    normed_pred_point_spherical = torch.cat((x_pred, y_pred_mu.unsqueeze(-1)), dim=-1).detach().numpy()
    normed_pred_point = spherical_to_cartisian(normed_pred_point_spherical)
    normed_pred_var = y_pred_var.detach().numpy()
    pred_point = normed_pred_point * points_std + points_mean
    pred_point_spherical = cartisian_to_spherical(pred_point).reshape(n_theta, n_phi, 3)
    pred_point_orn = R.from_euler('zy', pred_point_spherical[:,:,:2].reshape(-1,2))
    pred_var = normed_pred_var.reshape(n_theta, n_phi)
    # pred_point_rot_vec = pred_point_orn.apply(np.array([0,0,1])).reshape(n_theta, n_phi, 3)
    # pred_point_std = np.dot(pred_point_rot_vec, points_std)
    pred_var = pred_var * (points_std**2)
    plotter.plot_3d([pred_point.reshape(-1,3)], 'gp shape', c=pred_var.flatten(), true_aspect=True)

    # evaluate the shape
    # enumerate all possible orientations
    sec_starts = np.zeros((n_theta, n_phi, 20))
    sec_err_mu = -np.ones((n_theta, n_phi, 20)) # ndarray(n_orientation, n_section)
    sec_err_var = (1e-6)*np.ones((n_theta, n_phi, 20))
    obj_err_mu = np.zeros((n_theta, n_phi)) 
    obj_err_var = np.zeros((n_theta, n_phi))
    for i in trange(n_theta, disable=silent):
      for j in range(n_phi):
        rot = R.from_euler('zy', -pred_point_spherical[i,j,:2])
        pred_point_rot = (rot.apply(pred_point)).reshape(n_theta, n_phi, 3)
        pred_point_spherical_rot = cartisian_to_spherical(pred_point_rot)
        pred_point_rot = spherical_to_cartisian(pred_point_spherical_rot)
        # evaluate different sections
        for section_phi_idx, section_start in enumerate(np.arange(np.min(pred_point_rot[:,:,2]), np.max(pred_point_rot[:,:,2]), section_width)):
          sec_starts[i,j,section_phi_idx] = section_start
          section_end = section_start + section_width
          mask = (pred_point_rot[:,:,2] > section_start) & (pred_point_rot[:,:,2] < section_end)
          phi = pred_point_spherical_rot[mask, 1]
          theta = pred_point_spherical_rot[mask, 0]
          section_mu = pred_point_spherical_rot[mask, 2] * np.cos(phi)
          section_var = pred_var[mask] * (np.cos(phi)**2)
          hole_shape_polar= np.stack([theta, hole_shape_interp(theta)], axis=1)
          err_mu, err_var = eval_section(section_mu, section_var, hole_shape_polar)
          sec_err_mu[i, j, section_phi_idx] = err_mu
          sec_err_var[i, j, section_phi_idx] = err_var
        obj_mask = sec_err_var[i,j,:] > 1e-6
        obj_mu, obj_var = eval_shape(sec_err_mu[i,j][obj_mask], sec_err_var[i,j][obj_mask])
        obj_err_mu[i, j] = obj_mu
        obj_err_var[i, j] = obj_var
    all_ori = pred_point_spherical[:,:,:2]
    mll_ori_idx = np.unravel_index(np.argmax(obj_err_mu), obj_err_mu.shape)
    mll_ori = all_ori[mll_ori_idx]
    mll_ori_err_mu = obj_err_mu[mll_ori_idx]
    mll_ori_err_var = obj_err_var[mll_ori_idx]
    min_dist_mu.append(mll_ori_err_mu)
    min_dist_var.append(mll_ori_err_var)
    # plotter.plot_3d([data.reshape(-1,3)], f'object errors, \n best_ori={mll_ori}, \n mu={mll_ori_err_mu:.2f}, \n var={mll_ori_err_var:.2f}', c=obj_err_var.flatten(), axis_name=['theta', 'phi', 'margin'])
    data = np.concatenate([all_ori, np.expand_dims(obj_err_mu, axis=-1)], axis=-1)
    plotter.plot_heat(data, f'object errors mean, \n best_ori={mll_ori}, \n mu={mll_ori_err_mu:.2f}', axis_name=['theta', 'phi'])
    data = np.concatenate([all_ori, np.expand_dims(obj_err_var, axis=-1)], axis=-1)
    plotter.plot_heat(data, f'object errors var, \n var={mll_ori_err_var:.2f}', axis_name=['theta', 'phi'])

    # surrogate function
    obj_aq = obj_err_mu * 1.0 + obj_err_var * UCB_alpha
    max_pos = np.unravel_index(np.argmax(obj_aq), obj_aq.shape)
    best_ori = all_ori[max_pos]
    # data = np.concatenate([all_ori, np.expand_dims(obj_aq, axis=-1)], axis=-1)
    # plotter.plot_3d([data.reshape(-1,3), data[max_pos].reshape(-1,3)], f'aquisition function, \n best_ori={best_ori}', axis_name=['theta', 'phi', 'aq_value'])
    data = np.concatenate([all_ori, np.expand_dims(obj_aq, axis=-1)], axis=-1)
    plotter.plot_heat(data, f'object aq fn, \n best_ori={best_ori}', axis_name=['theta', 'phi'])

    # evaluate the section
    best_sec_err_mu = sec_err_mu[max_pos]
    best_sec_err_var = sec_err_var[max_pos]
    best_sec_start = sec_starts[max_pos]
    best_sec_mask = best_sec_err_var > 1e-5
    best_sec_err_mu = best_sec_err_mu[best_sec_mask]
    best_sec_err_var = best_sec_err_var[best_sec_mask]
    best_sec_range = best_sec_start[best_sec_mask] + section_width/2
    data = np.stack([best_sec_range, -best_sec_err_mu], axis=1)
    plotter.plot_2d([data], f'-section errors', c=best_sec_err_var*UCB_alpha, axis_name=['section displacement', 'margin'])
    sec_aq = -best_sec_err_mu * 1.0 + best_sec_err_var * UCB_alpha
    max_pos = np.argmax(sec_aq)
    best_sec_disp = best_sec_range[max_pos]
    data = np.stack([best_sec_range, sec_aq], axis=1)
    plotter.plot_2d([data, data[[max_pos]]], f'section aquisition function, \n best_sec_phi={best_sec_disp}', axis_name=['phi', 'aq_value'])

    # next exploration part
    if explore_policy == 'random':
      next_explore_pose = np.array([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-0.05, 0.05)])
    elif explore_policy == 'bo':
      next_explore_pose = np.array([best_ori[0], best_ori[1], best_sec_disp])
    explore_section_pos.append(next_explore_pose)
    # explore_section_pos.append()

    if mll_ori_err_mu > stop_bar:
      print('error reach the bar, stop exploration.')
      break
    # if len(min_dist_mu) > 2:
    #   if np.abs(min_dist_mu[-2] - min_dist_mu[-1]) < 0.0005:
    #     print('the gain from last sample is small, stop exploration')
    #     break
      # if min_dist_mu[-1] > -0.16:
      #   print('the gain from last sample is small, stop exploration')
      #   break

  plotter.row += 1

  data = np.stack((np.arange(len(min_dist_mu)), min_dist_mu), axis=-1)
  plotter.plot_2d([data], title='error change over time', c=np.array(min_dist_var))

  # plot final insert orientation
  max_pos = np.unravel_index(np.argmax(obj_err_mu), obj_err_mu.shape)
  best_ori = all_ori[max_pos]
  rot = R.from_euler('zy', -best_ori)
  insert_points=  rot.apply(points_dense)
  hole_3d = np.concatenate([hole_shape, np.zeros((hole_shape.shape[0],1))],axis=-1)
  final_err = true_object_err[max_pos]
  plotter.plot_3d([hole_3d, insert_points], title=f'final inseration orientation={best_ori}, \n err={final_err}', true_aspect=True, alpha=[1,0.05])

  plotter.save(f'../test/results/{object_name}_{explore_policy}_{UCB_alpha}_err{final_err:.2f}_step{explore_step+1}_init{init_explore_section[0]:.2f}_{init_explore_section[1]:.2f}_{init_explore_section[2]:.2f}.png')

  return final_err, explore_step+1 

if __name__ == '__main__':
  run_bo(object_name='Shape1', UCB_alpha=50000, if_plot=True, silent=False, max_explore_time=10)