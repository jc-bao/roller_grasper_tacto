import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import gpytorch
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from tqdm import trange
import open3d as o3d
from scipy.optimize import curve_fit
import seaborn as sns

def get_sphere_pcd(x_scale,y_scale,z_scale, density=21):
  theta, phi = np.meshgrid(np.linspace(-np.pi,np.pi,density), np.linspace(-np.pi/2,np.pi/2,density))
  x = np.cos(phi) * np.cos(theta) * x_scale
  y = np.cos(phi) * np.sin(theta) * y_scale
  z = np.sin(phi) * z_scale
  return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def get_cylinder_pcd(x_scale, y_scale, z_scale, density=21):
  theta, z = np.meshgrid(np.linspace(-np.pi, np.pi, density), np.linspace(-z_scale, z_scale, density))
  x = np.cos(theta) * x_scale
  y = np.sin(theta) * y_scale
  side = np.stack([x, y, z], axis=-1).reshape(-1, 3)
  xx, yy = np.meshgrid(np.linspace(-1, 1, density//3), np.linspace(-1, 1, density//3))
  rr = np.sqrt(xx**2 + yy**2)
  mask = rr<1
  circle = np.stack([xx[mask], yy[mask], np.zeros(np.sum(mask))], axis=-1)
  circle[:, 0] *= x_scale
  circle[:, 1] *= y_scale
  circle_up = circle.copy()
  circle_down = circle.copy()
  circle_up[:,-1] = z_scale
  circle_down[:,-1] = -z_scale
  return np.concatenate(
    [side, circle_up, circle_down], axis=0
  )

def get_cone_pcd(x_scale, y_scale, z_scale, density=21):
  theta, z = np.meshgrid(np.linspace(-np.pi, np.pi, density), np.linspace(-1, 1, density))
  x = np.cos(theta) * (-z+1)/2
  y = np.sin(theta) * (-z+1)/2
  side = np.stack([x, y, z], axis=-1).reshape(-1, 3)
  xx, yy = np.meshgrid(np.linspace(-1, 1, density//2), np.linspace(-1, 1, density//3))
  rr = np.sqrt(xx**2 + yy**2)
  mask = rr<1
  circle = np.stack([xx[mask], yy[mask], -np.ones(np.sum(mask))], axis=-1)
  pt = np.concatenate(
    [side, circle], axis=0
  )
  pt[:,0] *= x_scale
  pt[:,1] *= y_scale
  pt[:,2] *= z_scale
  return pt

def get_cube_pcd(x_scale, y_scale, z_scale, density=11):
  x, y, z = np.meshgrid(np.linspace(-1, 1, density), np.linspace(-1, 1, density), np.linspace(-1, 1, density))
  pt = np.stack([x, y, z], axis=-1).reshape(-1, 3)
  mask = np.abs(pt).max(axis=-1) == 1
  pt = pt[mask]
  pt[:, 0] *= x_scale
  pt[:, 1] *= y_scale
  pt[:, 2] *= z_scale
  return pt

def get_pyramid_pcd(x_scale, y_scale, z_scale, density=11):
  x, y, z = np.meshgrid(np.linspace(-1, 1, density), np.linspace(-1, 1, density), np.linspace(-1, 1, density))
  pt = np.stack([x, y, z], axis=-1).reshape(-1, 3)
  mask = np.abs(pt).max(axis=-1) == 1
  pt = pt[mask]
  pt[:, 0] *= x_scale
  pt[:, 1] *= y_scale
  pt[:, 2] *= z_scale
  return pt

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

def polar_to_cartisian(points_polar):
  points = np.zeros(points_polar.shape)
  points[..., 0] = points_polar[..., 1] * np.cos(points_polar[..., 0])
  points[..., 1] = points_polar[..., 1] * np.sin(points_polar[..., 0])
  return points

class Plotter():
  def __init__(self, num_figs, if_plot = True) -> None:
    self.num_figs = num_figs
    self.row = 1
    self.current_fig_id = np.ones(num_figs[0]+1, dtype=np.int)
    self.if_plot = if_plot
    if if_plot:
      self.fig = plt.figure(figsize=(8*num_figs[1],6*num_figs[0]))

  def create_ax(self, projection=None):
    if not self.if_plot:
      return
    fig_id = self.num_figs[1] * (self.row-1) + self.current_fig_id[self.row]
    ax = self.fig.add_subplot(*self.num_figs,fig_id, projection=projection)
    self.current_fig_id[self.row] += 1
    return ax

  def plot_2d(self, datas, title, c = None, axis_name = ['x', 'y']):
    if not self.if_plot:
      return
    ax = self.create_ax()
    for data in datas:
      if c is None:
        ax.plot(data[:,0], data[:,1])
        # ax.set_box_aspect(np.ptp(data[:,0])/np.ptp(data[:,1]))
      else:
        ax.plot(data[:,0], data[:,1])
        ax.fill_between(data[:,0], data[:,1]-c, data[:,1]+c, alpha=0.2)
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_title(title)

  def plot_heat(self, data, title, axis_name = ['x', 'y']):
    if not self.if_plot:
      return
    ax = self.create_ax()
    data = np.swapaxes(data, 0, 1)
    xlabels = ['{:,.2f}'.format(x) for x in data[0,:,0]]
    ylabels = ['{:,.2f}'.format(y) for y in data[:,0,1]]
    g = sns.heatmap(data[...,2], xticklabels=xlabels, yticklabels=ylabels, square=True, ax=ax, annot=False, fmt=".2f", cmap="YlGnBu")
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_title(title)
    g.set_xticklabels(xlabels)
    g.set_yticklabels(ylabels)

  def plot_polar(self, data, title, c = None):
    if not self.if_plot:
      return
    ax = self.create_ax('polar')
    if c is None:
      ax.plot(data[:,0], data[:,1])
    else:
      cm = plt.cm.get_cmap('viridis')
      sc = ax.scatter(data[:,0], data[:,1], s=10, cmap=cm, c=c)
      plt.colorbar(sc)
    ax.set_title(title)

  def plot_3d(self, datas, title, c = None, axis_name = ['x', 'y', 'z'], true_aspect=False, plane_pose = None, alpha=None):
    if not self.if_plot:
      return
    ax = self.create_ax('3d')
    if alpha is None:
      alpha = np.ones(len(datas))
    for i, data in enumerate(datas):
      if c is None:
        ax.scatter(data[:,0], data[:,1], data[:,2], alpha=alpha[i])
      else:
        cm = plt.cm.get_cmap('viridis')
        sc = ax.scatter(data[:,0], data[:,1], data[:,2], s=10, cmap=cm, c=c, alpha=alpha[i])
        self.cb = plt.colorbar(sc)
    if plane_pose is not None:
      xx, yy = np.meshgrid(np.linspace(-0.1, 0.1, 10), np.linspace(-0.1, 0.1, 10))
      zz = plane_pose[-1]*np.ones_like(xx)
      plane = np.stack((xx,yy,zz),axis=-1)
      plane_shape = plane.shape
      plane_orn = R.from_euler('zy', plane_pose[:2])
      plane_rot = plane_orn.apply(plane.reshape(-1,3)).reshape(plane_shape)
      ax.plot_surface(plane_rot[...,0], plane_rot[...,1], plane_rot[...,2], color='r', alpha=0.2)
    ax.set_xlabel(axis_name[0])
    ax.set_ylabel(axis_name[1])
    ax.set_zlabel(axis_name[2])
    ax.set_title(title)
    if true_aspect:
      ax.set_box_aspect((np.ptp(data[:,0]), np.ptp(data[:,1]), np.ptp(data[:,2])))

  def save(self, path):
    if not self.if_plot:
      return
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
  # hull = ConvexHull(points_projected)
  angle_step = np.pi/60
  angle = np.arange(-np.pi, np.pi, angle_step)
  points_polar = cartisian_to_polar(points_projected)
  points_angle_grid = (points_polar[:,0]//angle_step)*angle_step
  filtered_points_polar = []
  for a in angle:
    mask = np.abs(points_angle_grid - a)<1e-4
    if mask.sum() > 2:
      filtered_points_polar.append(np.array([a, points_polar[mask,1].max()]))
  filtered_points_polar = np.array(filtered_points_polar)
  filtered_points_cartesian = polar_to_cartisian(filtered_points_polar)
  return filtered_points_cartesian, filtered_points_polar

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
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess():
  def __init__(self, train_x, train_y, train_num = 100):
    # expand data to make the data periodic
    train_x_full = []
    train_y_full = []
    for delta_theta in [-torch.pi*2, 0, torch.pi*2]:
      train_x_full.append(train_x + torch.tensor([delta_theta, 0]))
      train_y_full.append(train_y)
    train_x = torch.cat(train_x_full, dim=0)
    train_y = torch.cat(train_y_full, dim=0)
    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    self.model = ExactGPModel(train_x, train_y, self.likelihood)
    hypers = {
      'covar_module.base_kernel.lengthscale': torch.tensor(0.3), # 1.5 for sphere
      # 'covar_module.kernels.1.base_kernel.lengthscale': torch.tensor(0.5),
    }
    self.model_params = self.model.initialize(**hypers)

    self.model.train()
    self.likelihood.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
    for i in (pbar := trange(train_num)):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = self.model(train_x)
      loss = -mll(output, train_y)
      loss.backward()
      pbar.set_description(f'Iter {i+1}/{train_num} - Loss: {loss.item():.3f}, lengthscale0: {self.model.covar_module.base_kernel.lengthscale.item():.3f}')
        # lengthscale1: {self.model.covar_module.kernels[1].base_kernel.lengthscale.item():.3f}')
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

class thinPlateModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
    super(thinPlateModel, self).__init__(train_x, train_y, likelihood)
    # self.mean_module = gpytorch.means.ZeroMean() 
    #self.covar_module = gpytorch.kernels.ScaleKernel(ThinPlateRegularizer(), outputscale_constraint=gpytorch.constraints.Interval(1e-5,1e-3))
    # self.covar_module = ThinPlateRegularizer()
    self.mean_module = gpytorch.means.ConstantMean()
    # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # self.covar_module = gpytorch.kernels.ScaleKernel(ThinPlateRegularizer())
    self.covar_module = ThinPlateRegularizer()


  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ThinPlateRegularizer(gpytorch.kernels.Kernel):
  # the sinc kernel is stationary
  is_stationary = True

  # We will register the parameter when initializing the kernel
  def __init__(self, dist_prior=None, dist_constraint=None, **kwargs):
    super().__init__(**kwargs)

    # register the raw parameter
    self.register_parameter(
      name='max_dist', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
    )

    # set the parameter constraint to be positive
    if dist_constraint is None:
      dist_constraint = gpytorch.constraints.GreaterThan(0.20)

    # register the constraint
    self.register_constraint("max_dist", dist_constraint)

    # set the parameter prior, see
    # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
    if dist_prior is not None:
      self.register_prior(
        "dist_prior",
        dist_prior,
        lambda m: m.length,
        lambda m, v: m._set_length(v),
      )

  # now set up the 'actual' paramter
  @property
  def maxdist(self):
    # when accessing the parameter, apply the constraint transform
    return self.raw_dist_constraint.transform(self.max_dist)

  @maxdist.setter
  def maxdist(self, value):
    return self._set_maxdist(value)

  def _set_maxdist(self, value):
    if not torch.is_tensor(value):
      value = torch.as_tensor(value).to(self.max_dist)
    # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
    self.initialize(max_dist=self.raw_dist_constraint.inverse_transform(value))

  # this is the kernel function
  def forward(self, x1, x2, diag=False, **params):
    # calculate the distance between inputs
    diff = self.covar_dist(x1, x2, diag=diag, **params)
    # prevent divide by 0 errors
    diff.where(diff == 0, torch.as_tensor(1e-20))
    # noise = 1e-5
    # white = noise*torch.eye(diff.shape[0], diff.shape[1])
    tp = 2*torch.pow(diff, 3)-3*self.max_dist * \
      torch.pow(diff, 2)+self.max_dist**3
    return tp

class GPIS():
  def __init__(self, train_x, train_y, train_num = 100):
    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
    self.model = thinPlateModel(train_x, train_y, self.likelihood)
    hypers = {
      'likelihood.noise_covar.noise': 0.1,
      'covar_module.max_dist': torch.tensor(5.0),
    }
    self.model_params = self.model.initialize(**hypers)

    self.model.train()
    self.likelihood.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
    for i in (pbar := trange(train_num)):
      # Zero gradients from previous iteration
      optimizer.zero_grad()
      # Output from model
      output = self.model(train_x)
      loss = -mll(output, train_y)
      loss.backward()
      pbar.set_description(f'Iter {i+1}/{train_num} - Loss: {loss.item():.3f}, lengthscale0: {self.model.covar_module.max_dist.item():.3f}, noise: {self.likelihood.noise_covar.noise.item():.3f}')
        # lengthscale1: {self.model.covar_module.kernels[1].base_kernel.lengthscale.item():.3f}')
      optimizer.step()

  def predict(self, step = np.pi/60):
    self.model.eval()
    self.likelihood.eval()

    # Make predictions by feeding model through likelihood
    theta = torch.arange(-np.pi, np.pi, step)
    phi = torch.arange(-np.pi/2, np.pi/2, step)
    r = torch.arange(0.1, 2, 0.01)
    theta, phi, r = torch.meshgrid((theta, phi, r))

    xyz = torch.stack((r*torch.cos(phi)*torch.cos(theta), r*torch.cos(phi)*torch.sin(theta), r*torch.sin(phi)), dim=-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
      observed_pred = self.likelihood(self.model(xyz))

    is_mean = observed_pred.mean
    is_var = observed_pred.variance

    is_min = torch.min(torch.abs(is_mean), dim=-1)[0]
    mask = (torch.abs(is_mean) == is_min.unsqueeze(-1))
    
    x_pred = torch.stack((theta[mask], phi[mask]), dim=-1)
    y_pred_mean = r[mask]
    y_pred_var = is_var[mask]

    return x_pred, y_pred_mean, y_pred_var


if __name__ == '__main__':
  plotter = Plotter(num_figs=1)
  points = get_cube_pcd(0.15, 0.1, 0.05)
  plotter.plot_3d([points], axis_name=['x', 'y', 'z'], title='cylinder')
  plt.show()