import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from tqdm import trange
import open3d as o3d
from scipy.optimize import curve_fit

def cartisian_to_spherical(points):
  points_spherical = np.zeros(points.shape)
  points_spherical[..., 0] = np.arctan2(points[..., 1], points[..., 0])
  points_spherical[..., 1] = np.arctan2(points[...,2], np.linalg.norm(points[...,:2], axis=-1))
  points_spherical[..., 2] = np.linalg.norm(points, axis=-1)
  return points_spherical

def spherical_to_cartisian(points_spherical):
  points = np.zeros(points_spherical.shape)
  points[..., 0] = points_spherical[..., 2] * np.cos(points_spherical[..., 1]) * np.cos(points_spherical[..., 0])
  points[..., 1] = points_spherical[..., 2] * np.cos(points_spherical[..., 1]) * np.sin(points_spherical[..., 0])
  points[..., 2] = points_spherical[..., 2] * np.sin(points_spherical[..., 1])
  return points

def cartisian_to_polar(points):
  points_polar = np.zeros(points.shape)
  points_polar[..., 0] = np.arctan2(points[..., 1], points[..., 0])
  points_polar[..., 1] = np.linalg.norm(points[..., :2], axis=-1)
  return points_polar

class Plotter():
  def __init__(self, num_figs) -> None:
    self.num_figs = num_figs
    self.current_fig_id = 1
    self.fig = plt.figure(figsize=(5, 5*num_figs))

  def plot_2d(self, datas, title, c = None, axis_name = ['x', 'y']):
    ax = self.fig.add_subplot(self.num_figs,1,self.current_fig_id)
    self.current_fig_id += 1
    for data in datas:
      if c is None:
        ax.plot(data[:,0], data[:,1])
        # ax.set_box_aspect(np.ptp(data[:,0])/np.ptp(data[:,1]))
      else:
        cm = plt.cm.get_cmap('viridis')
        sc = ax.scatter(data[:,0], data[:,1], s=10, cmap=cm, c=c)
        plt.colorbar(sc)
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_title(title)

  def plot_polar(self, data, title, c = None):
    ax = self.fig.add_subplot(self.num_figs,1,self.current_fig_id, projection='polar')
    self.current_fig_id += 1
    if c is None:
      ax.plot(data[:,0], data[:,1])
    else:
      cm = plt.cm.get_cmap('viridis')
      sc = ax.scatter(data[:,0], data[:,1], s=10, cmap=cm, c=c)
      plt.colorbar(sc)
    ax.set_title(title)

  def plot_3d(self, datas, title, c = None, axis_name = ['x', 'y', 'z'], true_aspect=False):
    ax = self.fig.add_subplot(self.num_figs,1,self.current_fig_id, projection='3d')
    self.current_fig_id += 1
    for data in datas:
      if c is None:
        ax.scatter(data[:,0], data[:,1], data[:,2])
      else:
        cm = plt.cm.get_cmap('viridis')
        sc = ax.scatter(data[:,0], data[:,1], data[:,2], s=10, cmap=cm, c=c)
        plt.colorbar(sc)
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_zlabel(axis_name[2])
    ax.set_title(title)
    if true_aspect:
      ax.set_box_aspect((np.ptp(data[:,0]), np.ptp(data[:,1]), np.ptp(data[:,2])))

  def plot_3d_surface(self, datas, title, c = None, axis_name = ['x', 'y', 'z'], true_aspect=False):
    ax = self.fig.add_subplot(self.num_figs,1,self.current_fig_id, projection='3d')
    def function(data, a, b, c, d, e, f, g, h, i, j, k):
      x = data[..., 0]
      y = data[..., 1]
      return a+b*x+c*y+d*x*x+e*y*y+f*x*x*x+g*y*y*y+h*x*x*x*x+i*y*y*y*y+j*x*x*x*x*x+k*y*y*y*y*y
    for data in datas:
      model_x_data = np.linspace(min(data[:,0]), max(data[:,1]), 100)
      model_y_data = np.linspace(min(data[:,0]), max(data[:,1]), 100)
      # create coordinate arrays for vectorized evaluations
      X, Y = np.meshgrid(model_x_data, model_y_data)
      XY = np.stack([X, Y], axis=2)
      # calculate Z coordinate array
      parameters, covariance = curve_fit(function, data[:,:2], data[:,2])
      Z = function(XY, *parameters)
      self.current_fig_id += 1
      if c is None:
        ax.plot_surface(X, Y, Z)
      else:
        # cm = plt.cm.get_cmap('viridis')
        parameters, covariance = curve_fit(function, data[:,:2], c)
        C = function(XY, *parameters)
        sc = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(C))
        plt.colorbar(sc)
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_zlabel(axis_name[2])
    ax.set_title(title)
    if true_aspect:
      ax.set_box_aspect((np.ptp(data[:,0]), np.ptp(data[:,1]), np.ptp(data[:,2])))

  def save(self, path):
    self.fig.savefig(path)
    plt.close(self.fig)

'''get the data'''
def load_data(path, voxel_size=0.02):
  origin_pcd = o3d.io.read_point_cloud(path)
  pcd = origin_pcd.voxel_down_sample(voxel_size=voxel_size)
  points = np.asarray(pcd.points)
  return points

def get_hole_shape(points, orn:R):
  points_transformed = orn.apply(points)
  points_projected = points_transformed[:, :2]
  # get convex hull
  hull = ConvexHull(points_projected)
  return points_projected[hull.vertices]

def get_section_points(points, section_poses, section_width = 0.05):
  mask_in = np.zeros(points.shape[0], dtype=bool)
  for section_pos in section_poses:
    orn = R.from_euler('xyz', np.append(section_pos[:2],0))
    disp = section_pos[2]
    z_start = disp - section_width/2
    z_end = disp + section_width/2
    points_transformed = orn.apply(points)
    mask_in |= ((points_transformed[:, 2] > z_start) & (points_transformed[:, 2] < z_end))
  return points[mask_in]

'''probabilistic model'''
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.RBFKernel())

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess():
  def __init__(self, train_x, train_y, train_num = 100):
    # expand data to make the data periodic
    train_x_full = []
    train_y_full = []
    for delta_theta in [-torch.pi, 0, torch.pi]:
      train_x_full.append(train_x + torch.tensor([delta_theta, 0]))
      train_y_full.append(train_y)
    train_x = torch.cat(train_x_full, dim=0)
    train_y = torch.cat(train_y_full, dim=0)
    self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.tensor([0.0]))
    self.model = ExactGPModel(train_x, train_y, self.likelihood)
    hypers = {
      'covar_module.kernels.0.base_kernel.lengthscale': torch.tensor(0.4),
      'covar_module.kernels.1.base_kernel.lengthscale': torch.tensor(0.02),
    }
    self.model_params = self.model.initialize(**hypers)

    self.model.train()
    self.likelihood.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
    for i in (pbar := trange(train_num)):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = self.model(train_x)
      loss = -mll(output, train_y)
      loss.backward()
      pbar.set_description(f'Iter {i+1}/{train_num} - Loss: {loss.item():.3f}, lengthscale0: {self.model.covar_module.kernels[0].base_kernel.lengthscale.item():.3f}, lengthscale1: {self.model.covar_module.kernels[1].base_kernel.lengthscale.item():.3f}')
      optimizer.step()

  def predict(self, step = np.pi/60):
    self.model.eval()
    self.likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
      theta = torch.arange(-np.pi, np.pi, step, dtype=torch.float64)
      phi = torch.arange(-np.pi/2, np.pi/2, step, dtype=torch.float64)
      x_pred = torch.stack(torch.meshgrid(theta, phi), dim=-1).view(-1, 2)
      observed_pred = self.likelihood(self.model(x_pred))

    return x_pred, observed_pred.mean, observed_pred.variance