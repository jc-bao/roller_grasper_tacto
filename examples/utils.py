from this import d
import pybullet as pb
import pybulletX as px
import collections
import numpy as np
import gym
from PIL import Image, ImageFont, ImageDraw
import open3d as o3d
from typing import List

import pybullet as p


class Camera:
  def __init__(self, width: int = 320, height: int = 240, target: List[float] = [0., 0., 0.1], dist: float = 0.1, yaw: float = 90.0, pitch: float = -20.0, roll: float = 0, fov: float = 90.0, near: float = 0.001, far: float = 100.0, crop: List[float] = [1,1]):
    self.width = width
    self.height = height
    self.x_start = int(width * (1 - crop[0]) / 2)
    self.y_start = int(height * (1 - crop[1]) / 2)
    self.x_end = int(width * (1 + crop[0]) / 2)
    self.y_end = int(height * (1 + crop[1]) / 2)

    self.camTargetPos = target
    self.camDistance = dist
    self.upAxisIndex = 2

    self.yaw = yaw
    self.pitch = pitch
    self.roll = roll

    self.near = near
    self.far = far

    self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
      self.camTargetPos, self.camDistance, self.yaw, self.pitch, self.roll, self.upAxisIndex
    )

    aspect = width / height
    self.projectionMatrix = p.computeProjectionMatrixFOV(
      fov, aspect, near, far
    )

  def update_dist(self, dist:float):
    self.camDistance = dist
    self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
      self.camTargetPos, self.camDistance, self.yaw, self.pitch, self.roll, self.upAxisIndex
    )
    
  def get_image(self):
    img_arr = p.getCameraImage(
      self.width,
      self.height,
      self.viewMatrix,
      self.projectionMatrix,
      shadow=1,
      lightDirection=[1, 1, 1],
      renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgb = np.asarray(img_arr[2], dtype=np.uint8)[self.y_start:self.y_end, self.x_start:self.x_end]  # color image RGB H x W x 3 (uint8)
    dep = np.asarray(img_arr[3], dtype=np.uint8)[self.y_start:self.y_end, self.x_start:self.x_end]  # depth image H x W (float32)
    dep = self.far * self.near / \
      (self.far - (self.far - self.near) * dep)
    return rgb, dep


def _get_dtype_min_max(dtype):
  if np.issubdtype(dtype, np.integer):
    return np.iinfo(dtype).min, np.iinfo(dtype).max
  if np.issubdtype(dtype, np.floating):
    return np.finfo(dtype).min, np.finfo(dtype).max
  raise NotImplementedError


def convert_obs_to_obs_space(obs):
  if isinstance(obs, (int, float)):
    return convert_obs_to_obs_space(np.array(obs))

  if isinstance(obs, np.ndarray):
    min_, max_ = _get_dtype_min_max(obs.dtype)
    return gym.spaces.Box(low=min_, high=max_, shape=obs.shape, dtype=obs.dtype)

  # for list-like container
  if isinstance(obs, (list, tuple)):
    if np.all([isinstance(_, float) for _ in obs]):
      return convert_obs_to_obs_space(np.array(obs))
    return gym.spaces.Tuple([convert_obs_to_obs_space(_) for _ in obs])

  # for any dict-like container
  if isinstance(obs, collections.abc.Mapping):
    # SpaceDict inherits from gym.spaces.Dict and provides more functionalities
    return px.utils.SpaceDict({k: convert_obs_to_obs_space(v) for k, v in obs.items()})


def unifrom_sample_quaternion():
  q = p.getQuaternionFromEuler([np.random.uniform(-np.pi, np.pi),
                               np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)])
  return q


def char_to_pixels(text, path='arialbd.ttf', fontsize=14):
  # font = ImageFont.truetype(path, fontsize)
  # w, h = font.getsize(text)
  # h *= 2
  image = Image.new('L', (64, 16), 256)
  draw = ImageDraw.Draw(image)
  draw.text((0, 0), text, align='center')
  arr = np.asarray(image)
  arr = np.expand_dims(arr, 2).repeat(3, axis=2)
  return arr


def pairwise_registration(source, target, max_correspondence_distance_coarse,
                          max_correspondence_distance_fine):
  icp_coarse = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance_coarse, np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
  icp_fine = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance_fine,
    icp_coarse.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
  transformation_icp = icp_fine.transformation
  rmse, fitness = icp_fine.inlier_rmse, icp_fine.fitness
  information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
    source, target, max_correspondence_distance_fine,
    icp_fine.transformation)
  return transformation_icp, information_icp, rmse, fitness


def create_o3dvis(render_size: int = 512):
  o3dvis = o3d.visualization.Visualizer()
  o3dvis.create_window(
    width=render_size, height=render_size, visible=False)
  o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
  o3dvis.get_view_control().set_front(np.array([1, 1, 1]))
  o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
  o3dvis.get_render_option().background_color = (0.5, 0.5, 0.5)
  return o3dvis


if __name__ == '__main__':
  arr = char_to_pixels('a')
  print(arr)
