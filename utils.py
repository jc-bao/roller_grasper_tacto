import pybullet as pb
import pybulletX as px
import collections
import numpy as np
import gym
from PIL import Image, ImageFont, ImageDraw
import open3d as o3d

import pybullet as p


class Camera:
  def __init__(self, width=320, height=240):
    self.width = width
    self.height = height

    camTargetPos = [0.5, 0, 0.05]
    camDistance = 0.4
    upAxisIndex = 2

    yaw = 90
    pitch = -30.0
    roll = 0

    fov = 60
    nearPlane = 0.01
    farPlane = 100

    self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
      camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
    )

    aspect = width / height

    self.projectionMatrix = p.computeProjectionMatrixFOV(
      fov, aspect, nearPlane, farPlane
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

    rgb = img_arr[2]  # color image RGB H x W x 3 (uint8)
    dep = img_arr[3]  # depth image H x W (float32)
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
  q = p.getQuaternionFromEuler([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)])
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

if __name__ == '__main__':
  arr = char_to_pixels('a')
  print(arr)