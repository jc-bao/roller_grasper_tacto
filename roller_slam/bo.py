import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import torch, gpytorch
import open3d as o3d 

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
  x_min = torch.linspace(-0.4, 0.4, 11).unsqueeze(-1)
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

def conduct_bo(errs):
  return np.max(errs.std())


def main():
  # object SLAM parameters
  n_phi = 10
  n_theta = n_phi * 2
  section_width = 0.02
  angle_step = torch.pi/n_phi

  plotter = Plotter(num_figs=32)
  # load the data
  new_pcd = o3d.io.read_point_cloud("../test/assets/objects/Shape1.ply")
  points = np.asarray(new_pcd.points)
  points = points/np.max(points,0)*0.1
  points_mean = np.mean(points, axis=0)
  points_std = np.std(points, axis=0)
  points_mean = 0
  points_std = 0.08
  points_normed = (points - points_mean) / points_std
  plotter.plot_3d([points], 'normed bottle point clouds', true_aspect=True)

  # get the hole shape
  hole_euler = np.array([np.pi/3, 0, 0])
  hole_shape, hole_shape_polar = get_hole_shape(points, R.from_euler('xyz', hole_euler))
  # extra_pts = []
  # for i in range(hole_shape.shape[0]):
  #   if i <= (hole_shape.shape[0]-2):
  #     pt = hole_shape[i:i+2]
  #   else:
  #     pt = hole_shape[[i, 0]]
  #   x = np.linspace(pt[0,0], pt[1,0], 10)[1:-1]
  #   y = pt[0,1] + (pt[1,1] - pt[0,1])/(pt[1,0] - pt[0,0]) * (x - pt[0,0])
  #   extra_pts.append(np.stack([x,y], axis=-1))
  # extra_pts = np.concatenate(extra_pts, axis=0)
  # hole_shape = np.concatenate([hole_shape, extra_pts], axis=0)
  # hole_shape_polar = cartisian_to_polar(hole_shape)
  # sort array
  # hole_shape_polar = hole_shape_polar[hole_shape_polar[:, 0].argsort()]
  # add extra points
  if hole_shape_polar[-1,0] - hole_shape_polar[0,0] < 2*np.pi: 
    left_point = hole_shape_polar[-1].copy()
    left_point[0] -= 2 * np.pi
    right_point = hole_shape_polar[0].copy()
    right_point[0] += 2 * np.pi
    hole_shape_polar = np.concatenate([[left_point], hole_shape_polar, [right_point]], axis=0)
  # interpolate the hole shape to the the same resolution as the evaluation grid
  hole_shape_interp = interp1d(hole_shape_polar[:, 0], hole_shape_polar[:, 1])
  hole_shape_polar= np.stack([np.linspace(-np.pi, np.pi, n_theta), hole_shape_interp(np.linspace(-np.pi, np.pi, n_theta))], axis=1)
  plotter.plot_2d([hole_shape], 'hole shape')
  plotter.plot_polar(hole_shape_polar, 'hole shape polar')

  # evaluate the object
  obj_err_mu = np.zeros((n_theta, n_phi))
  pred_point = points
  pred_point_spherical = cartisian_to_spherical(pred_point)
  pred_var = np.zeros(pred_point.shape[0])
  all_ori = np.stack(np.meshgrid(np.linspace(-np.pi, np.pi, n_theta), np.linspace(-np.pi/2, np.pi/2, n_phi)), axis=-1).transpose(1,0,2)
  for i in range(n_theta):
    for j in range(n_phi):
      rot = R.from_euler('zy', all_ori[i,j], degrees=False)
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
  plotter.plot_heat(data, 'object evaluation result', axis_name=['theta', 'phi'])

  explore_section_pos = [np.array([0, 0, 0])] # explore the center section first
  min_dist_mu = []
  min_dist_var = []
  for explore_step in range(0):
    # explore the object
    section_points = get_section_points(points, explore_section_pos, section_width)
    new_points = get_section_points(points, [explore_section_pos[-1]], section_width)
    plotter.plot_3d([new_points, section_points], f'section points, orn={explore_section_pos}', true_aspect=True)

    # fit a probabilistic model to the section points
    normed_section_points = (section_points - points_mean) / points_std
    normed_section_point_spherical = cartisian_to_spherical(normed_section_points)
    normed_section_point_tensor = torch.tensor(normed_section_point_spherical)
    gp_shape = GaussianProcess(normed_section_point_tensor[:,:2], normed_section_point_tensor[:,2], train_num=100)
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
    for i in range(n_theta):
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
        obj_mask = sec_err_var[i,j,:] > 1e-5
        obj_mu, obj_var = eval_shape(sec_err_mu[i,j][obj_mask], sec_err_var[i,j][obj_mask])
        obj_err_mu[i, j] = obj_mu
        obj_err_var[i, j] = obj_var
    all_ori = pred_point_spherical[:,:,:2]
    data = np.concatenate([all_ori, np.expand_dims(obj_err_mu, axis=-1)], axis=-1)
    mll_ori_idx = np.unravel_index(np.argmax(obj_err_mu), obj_err_mu.shape)
    mll_ori = all_ori[mll_ori_idx]
    mll_ori_err_mu = obj_err_mu[mll_ori_idx]
    mll_ori_err_var = obj_err_var[mll_ori_idx]
    min_dist_mu.append(mll_ori_err_mu)
    min_dist_var.append(mll_ori_err_var)
    plotter.plot_3d([data.reshape(-1,3)], f'object errors, \n best_ori={mll_ori}, \n mu={mll_ori_err_mu:.2f}, \n var={mll_ori_err_var:.2f}', c=obj_err_var.flatten(), axis_name=['theta', 'phi', 'margin'])

    # surrogate function
    obj_aq = obj_err_mu * 1.0 + obj_err_var * 20
    max_pos = np.unravel_index(np.argmax(obj_aq), obj_aq.shape)
    best_ori = all_ori[max_pos]
    data = np.concatenate([all_ori, np.expand_dims(obj_aq, axis=-1)], axis=-1)
    plotter.plot_3d([data.reshape(-1,3), data[max_pos].reshape(-1,3)], f'aquisition function, \n best_ori={best_ori}', axis_name=['theta', 'phi', 'aq_value'])

    # evaluate the section
    best_sec_err_mu = sec_err_mu[max_pos]
    best_sec_err_var = sec_err_var[max_pos]
    best_sec_start = sec_starts[max_pos]
    best_sec_mask = best_sec_err_var > 1e-5
    best_sec_err_mu = best_sec_err_mu[best_sec_mask]
    best_sec_err_var = best_sec_err_var[best_sec_mask]
    best_sec_range = best_sec_start[best_sec_mask] + section_width/2
    data = np.stack([best_sec_range, -best_sec_err_mu], axis=1)
    plotter.plot_2d([data], f'-section errors', c=best_sec_err_var, axis_name=['section displacement', 'margin'])
    sec_aq = -best_sec_err_mu*0 + best_sec_err_var * 50
    max_pos = np.argmax(sec_aq)
    best_sec_disp = best_sec_range[max_pos]
    data = np.stack([best_sec_range, sec_aq], axis=1)
    plotter.plot_2d([data, data[[max_pos]]], f'section aquisition function, \n best_sec_phi={best_sec_disp}', axis_name=['phi', 'aq_value'])

    # next exploration part
    explore_section_pos.append(np.append(best_ori, best_sec_disp))
    # explore_section_pos.append(np.array([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi/2, np.pi/2), np.random.uniform(-0.05, 0.05)]))

    if len(min_dist_mu) > 2:
      if np.abs(min_dist_mu[-2] - min_dist_mu[-1]) < 0.0005:
        print('the gain from last sample is small, stop exploration')
        break
      # if min_dist_mu[-1] > -0.16:
      #   print('the gain from last sample is small, stop exploration')
      #   break
  data = np.stack((np.arange(len(min_dist_mu)), min_dist_mu), axis=-1)
  plotter.plot_2d([data], title='error change over time', c=np.array(min_dist_var))

  # plot final insert orientation
  max_pos = np.unravel_index(np.argmax(obj_err_mu), obj_err_mu.shape)
  best_ori = all_ori[max_pos]
  rot = R.from_euler('zy', -best_ori)
  insert_points=  rot.apply(points)
  hole_3d = np.concatenate([hole_shape, np.zeros((hole_shape.shape[0],1))],axis=-1)
  plotter.plot_3d([hole_3d, insert_points], title=f'final inseration orientation={best_ori}', true_aspect=True)

  # evaluate different sections
  sec_starts = []
  sec_err = []
  for section_phi_idx, section_start in enumerate(np.arange(np.min(insert_points[:,2]), np.max(insert_points[:,2]), section_width)):
    sec_starts.append(section_start)
    section_end = section_start + section_width
    mask = (insert_points[:,2] > section_start) & (insert_points[:,2] < section_end)
    phi = insert_points[mask, 1]
    theta = insert_points[mask, 0]
    section_mu = insert_points[mask, 2] * np.cos(phi)
    hole_shape_polar= np.stack([theta, hole_shape_interp(theta)], axis=1)
    err_mu, err_var = eval_section(section_mu, np.zeros_like(section_mu), hole_shape_polar)
    sec_err.append(err_mu)

  data = np.stack([sec_starts, sec_err], axis=1)
  plotter.plot_2d([data], title='section error', c=np.zeros_like(sec_err), axis_name=['section start', 'error'])

  plotter.save('../test/results/bottle_BO.png')

if __name__ == '__main__':
  main()