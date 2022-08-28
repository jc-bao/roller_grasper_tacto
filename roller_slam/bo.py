import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import torch, gpytorch

from roller_slam.utils import load_data, get_hole_shape, get_section_points, GaussianProcess, Plotter, cartisian_to_spherical, spherical_to_cartisian, cartisian_to_polar

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

def eval_projection(projection_shape, hole_shape):
  return False

def eval_shape(sec_mu, sec_var):
  x_min = torch.linspace(-2, 2, 11).unsqueeze(-1)
  dist = torch.tensor(sec_mu)
  dist_std = torch.sqrt(torch.tensor(sec_var))
  x_normed = (x_min - dist) / dist_std # Tensor(n_x, n_theta)
  log_cdf = gpytorch.log_normal_cdf(x_normed)
  mask = log_cdf > -5
  cdf = torch.zeros_like(log_cdf)
  cdf[mask] = torch.exp(log_cdf[mask])
  cdf = torch.prod(1 - cdf, dim=1) # Tensor(n_x)
  pdf = cdf[:-1] - cdf[1:]
  pdf = pdf / pdf.sum()
  mean = torch.sum(pdf * x_min[:-1,0])
  var = torch.sum(pdf * ((x_min[:-1,0] - mean)**2))
  return mean, var

def eval_section(y_mu, y_var, hole_shape_polar):
  assert y_mu.shape[0] == hole_shape_polar.shape[
      0], "The number of points in the hole shape and the Gaussian Process must be the same"
  x_min = torch.linspace(-2, 2, 101).unsqueeze(-1)
  dist = torch.tensor(hole_shape_polar[:,1] - y_mu)
  dist_std = torch.sqrt(torch.tensor(y_var))
  x_normed = (x_min - dist) / dist_std # Tensor(n_x, n_theta)
  log_cdf = gpytorch.log_normal_cdf(x_normed)
  mask = log_cdf > -5
  cdf = torch.zeros_like(log_cdf)
  cdf[mask] = torch.exp(log_cdf[mask])
  cdf = torch.prod(1 - cdf, dim=1) # Tensor(n_x)
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
  angle_step = torch.pi/n_phi

  # num_plts = 3
  # fig = plt.figure(figsize=(5, 5*num_plts))
  plotter = Plotter(num_figs=16)
  # load the data
  points_normed = load_data('../test/assets/bottle.ply')
  plotter.plot_3d(points_normed, 'bottle point clouds')

  # get the hole shape
  hole_euler = np.array([0, 0, 0])
  hole_shape = get_hole_shape(points_normed, R.from_euler('xyz', hole_euler))
  hole_shape_polar = cartisian_to_polar(hole_shape)
  # sort array
  hole_shape_polar = hole_shape_polar[hole_shape_polar[:, 0].argsort()]
  # add extra points
  left_point = hole_shape_polar[-1].copy()
  left_point[0] -= 2 * np.pi
  right_point = hole_shape_polar[0].copy()
  right_point[0] += 2 * np.pi
  hole_shape_polar = np.concatenate([[left_point], hole_shape_polar, [right_point]], axis=0)
  # interpolate the hole shape to the the same resolution as the evaluation grid
  hole_shape_interp = interp1d(hole_shape_polar[:, 0], hole_shape_polar[:, 1], kind='cubic')
  hole_shape_polar= np.stack([np.linspace(-np.pi, np.pi, n_theta), hole_shape_interp(np.linspace(-np.pi, np.pi, n_theta))], axis=1)
  plotter.plot_2d(hole_shape, 'hole shape')
  plotter.plot_polar(hole_shape_polar, 'hole shape polar')

  # explore the object
  explore_section_pos = np.array([0,0,0]) # explore the center section first
  section_points = get_section_points(points_normed, explore_section_pos)
  plotter.plot_3d(section_points, f'section points, orn={explore_section_pos}')

  # fit a probabilistic model to the section points
  section_point_spherical = cartisian_to_spherical(section_points)
  section_point_tensor = torch.tensor(section_point_spherical)
  gp_shape = GaussianProcess(section_point_tensor[:,:2], section_point_tensor[:,2])
  x_pred, y_pred_mu, y_pred_var = gp_shape.predict(step=angle_step)
  pred_point_spherical = torch.cat((x_pred, y_pred_mu.unsqueeze(-1)), dim=-1).detach().numpy()
  pred_point = spherical_to_cartisian(pred_point_spherical)
  pred_var = y_pred_var.detach().numpy()
  plotter.plot_3d(pred_point.reshape(-1,3), 'gp shape', c=pred_var.flatten())

  # evaluate the shape
  # enumerate all possible orientations
  theta = np.arange(-np.pi, np.pi, angle_step)
  phi = np.arange(-np.pi/2, np.pi/2, angle_step)
  all_ori = np.stack(np.meshgrid(theta, phi), axis=-1).reshape(-1, 2)
  y_pred_mu_new = y_pred_mu.reshape(n_theta, n_phi).detach().numpy()
  y_pred_var_new = y_pred_var.reshape(n_theta, n_phi).detach().numpy()
  # try different object orientations
  obj_err_mu = np.zeros(n_theta*n_phi)
  obj_err_var = np.zeros(n_theta*n_phi)
  for i, ori in enumerate(all_ori):
    x_pred_new = (x_pred + ori).reshape(n_theta, n_phi, 2).detach().numpy()
    x_pred_new[:,:,0] = (x_pred_new[:,:,0]+np.pi) % (2 * np.pi) - np.pi
    x_pred_new[:,:,1] = (x_pred_new[:,:,1]+np.pi/2) % np.pi - np.pi/2
    theta_argsort = np.argsort(x_pred_new[:,:,0], axis=0)
    phi_argsort = np.argsort(x_pred_new[:,:,1], axis=1)
    y_pred_mu_ori = y_pred_mu_new[theta_argsort, phi_argsort]
    y_pred_var_ori = y_pred_var_new[theta_argsort, phi_argsort]
    # debug: visualize new shape
    # data = np.concatenate([x_pred.reshape(-1,2), np.expand_dims(y_pred_mu_ori.flatten(), axis=1)], axis=1)
    # data = spherical_to_cartisian(data)
    # plotter.plot_3d(data, f'gp shape, orn={ori}')
    # evaluate different sections
    sec_errs_mu, sec_errs_var = [], []
    for section_phi_idx in range(n_phi):
      section_y_mu = y_pred_mu_ori[:, section_phi_idx] * np.cos(phi[section_phi_idx])
      section_y_var = y_pred_var_ori[:, section_phi_idx] * (np.cos(phi[section_phi_idx])**2)
      err_mu, err_var = eval_section(section_y_mu, section_y_var, hole_shape_polar)
      sec_errs_mu.append(err_mu)
      sec_errs_var.append(err_var)
    sec_errs_mu = np.array(sec_errs_mu)
    sec_errs_var = np.array(sec_errs_var)
    obj_mu, obj_var = eval_shape(sec_errs_mu, sec_errs_var)
    obj_err_mu[i] = obj_mu
    obj_err_var[i] = obj_var
  data = np.concatenate([all_ori, np.expand_dims(obj_err_mu, axis=1)], axis=1)
  plotter.plot_3d(data, f'object errors, ori={ori}', c=obj_err_var, axis_name=['theta', 'phi', 'margin'])

  # surrogate function
  obj_aq = obj_err_mu + obj_err_var * 10
  max_pos = np.argmax(obj_aq)
  data = np.concatenate([all_ori, np.expand_dims(obj_aq, axis=1)], axis=1)
  plotter.plot_3d(data, f'surrogate function, ori={ori}', axis_name=['theta', 'phi', 'aq_value'])

  # find the best orientation
  best_ori_idx = np.argmax(obj_aq)

  # evaluate the shape
  # Question: use joint distribution v.s. use marginal distribution to evaluate model?
  # Answer: if we care about the whole object, then we need to use the joint distribution. If we only care about one point, then marginal distribution is fine. In this case, we use the joint distribution to evaluate the whole objects. 
  # Pipeline: 1. get the joint distribution of the object; 2. project this distribution into a orientation; 3. get the hull distribution of the projected object; 4. evaluate the shape of the hull distribution compared with the hole shape.
  # Question: do we need to calculate the exact joint distribution of object projection? The contour shape of the object can be very complex to present, which is a combination of skew normal distribution. 
  # Answer: we can evaluate each point's probability of lies in the boundry. 

  plotter.save('../test/results/bottle_BO.png')

if __name__ == '__main__':
  main()