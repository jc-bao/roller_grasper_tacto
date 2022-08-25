import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from tqdm import trange
import open3d as o3d

def cartisian_to_spherical(points):
  points_spherical = np.zeros(points.shape)
  points_spherical[:, 0] = np.arctan2(points[:, 1], points[:, 0])
  points_spherical[:, 1] = np.arctan2(points[:,2], np.linalg.norm(points[:,:2], axis=1))
  points_spherical[:, 2] = np.linalg.norm(points, axis=1)
  return points_spherical

class Plotter():
  def __init__(self, num_figs) -> None:
    self.num_figs = num_figs
    self.current_fig_id = 1
    self.fig = plt.figure(figsize=(5, 5*num_figs))

  def plot_sphere_coordinate(self, data, title):
    THETA, PHI, R = data[:,0], data[:,1], data[:,2]
    X = R * np.cos(PHI) * np.cos(THETA)
    Y = R * np.cos(PHI) * np.sin(THETA)
    Z = R * np.sin(PHI)
    ax = self.fig.add_subplot(self.num_figs, 1, self.current_fig_id, projection='3d')
    self.current_fig_id += 1
    plot = ax.scatter(X, Y, Z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))

  def plot_2d(self, data, title):
    ax = self.fig.add_subplot(self.num_figs,1,self.current_fig_id)
    self.current_fig_id += 1
    ax.plot(data[:,0], data[:,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_box_aspect(np.ptp(data[:,0])/np.ptp(data[:,1]))

  def plot_3d(self, data, title):
    ax = self.fig.add_subplot(self.num_figs,1,self.current_fig_id, projection='3d')
    self.current_fig_id += 1
    ax.scatter(data[:,0], data[:,1], data[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.set_box_aspect((np.ptp(data[:,0]), np.ptp(data[:,1]), np.ptp(data[:,2])))

  def save(self, path):
    self.fig.savefig(path)
    plt.close(self.fig)

'''get the data'''
def load_data(path, voxel_size=0.02):
  origin_pcd = o3d.io.read_point_cloud(path)
  pcd = origin_pcd.voxel_down_sample(voxel_size=voxel_size)
  points = np.asarray(pcd.points)
  mean = np.mean(points, axis=0)
  std = np.std(points, axis=0)
  return (points - mean) / std

def get_hole_shape(points, orn:R):
  points_transformed = orn.apply(points)
  points_projected = points_transformed[:, :2]
  # get convex hull
  hull = ConvexHull(points_projected)
  return points_projected[hull.vertices]

def get_section_points(points, section_pos, section_width = 0.5):
  orn = R.from_euler('xyz', np.append(section_pos[:2],0))
  disp = section_pos[2]
  z_start = disp - section_width/2
  z_end = disp + section_width/2
  points_transformed = orn.apply(points)
  return points_transformed[(points_transformed[:, 2] > z_start) & (points_transformed[:, 2] < z_end)]

'''probabilistic model'''
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(
      gpytorch.kernels.RBFKernel())

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess():
  def __init__(self, train_x, train_y, train_num = 100):
    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    self.model = ExactGPModel(train_x, train_y, self.likelihood)
    hypers = {
      'covar_module.base_kernel.lengthscale': torch.tensor(0.572),
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
      pbar.set_description(f'Iter {i+1}/{train_num} - Loss: {loss.item():.3f}, lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f}')
      optimizer.step()

  def predict(self, step = np.pi/60):
    self.model.eval()
    self.likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
      theta = torch.arange(-np.pi, np.pi, step, dtype=torch.float64)
      phi = torch.linspace(-np.pi/2, np.pi/2, step, dtype=torch.float64)
      x_pred = torch.stack(torch.meshgrid(theta, phi), dim=-1)
      observed_pred = self.likelihood(self.model(x_pred))

    y_pred_mean = observed_pred.mean.detach().numpy()
    y_pred_std = observed_pred.stddev.detach().numpy()

    return x_pred.detach().numpy(), y_pred_mean, y_pred_std

