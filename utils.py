import pybullet as pb
import pybulletX as px
import collections
import numpy as np
import gym
from PIL import Image, ImageFont, ImageDraw
import open3d as o3d
from attrdict import AttrDict
import functools
from gym import spaces
from pybulletX.utils import SpaceDict

import pybullet as p


class Camera:
  def __init__(self, cfg:AttrDict):
    self.cfg = cfg
    self.x_start = 
    self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
      self.cfg.camTargetPos, self.cfg.camDistance, self.cfg.yaw, self.cfg.pitch, self.cfg.roll, self.cfg.upAxisIndex
    )
    self.projectionMatrix = p.computeProjectionMatrixFOV(
      self.cfg.fov, self.cfg.width / self.cfg.height, self.cfg.nearPlane, self.cfg.farPlane
    )
  
  def update(self, cfg:AttrDict):
    self.cfg.update(**cfg)
    self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
      self.cfg.camTargetPos, self.cfg.camDistance, self.cfg.yaw, self.cfg.pitch, self.cfg.roll, self.cfg.upAxisIndex
    )
    self.projectionMatrix = p.computeProjectionMatrixFOV(
      self.cfg.fov, self.cfg.width / self.cfg.height, self.cfg.nearPlane, self.cfg.farPlane
    )

  def get_image(self):
    img_arr = p.getCameraImage(
      self.cfg.width,
      self.cfg.height,
      self.viewMatrix,
      self.projectionMatrix,
      shadow=1,
      lightDirection=[1, 1, 1],
      renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = img_arr[2]  # color image RGB H x W x 3 (uint8)
    dep = img_arr[3]  # depth image H x W (float32)
    return rgb, dep


class RollerGrapser(px.Robot):
  wrist_vel = 1
  pitch_vel = 1
  roll_vel = 1
  wrist_joint_name = "wrist"
  gripper_names = [
    "joint1_left",
    "joint1_right",
  ]
  pitch_names = [
    'joint3_left',
    'joint3_right'
  ]
  roll_names = [
    'joint4_left',
    'joint4_right'
  ]
  digit_joint_names = ["joint5_left", "joint5_right"]

  MAX_FORCES = 200

  def __init__(self, robot_params, init_state):
    super().__init__(**robot_params)

    self.zero_pose = self._states_to_joint_position(init_state)
    self.reset()

  @property
  @functools.lru_cache(maxsize=None)
  def state_space(self):
    return SpaceDict(
      {
        "wrist_angle": spaces.Box(
          low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        ),
        "finger_l": spaces.Box(
          low=0, high=0.1, shape=(1,), dtype=np.float32
        ),
        "finger_r": spaces.Box(
          low=0, high=0.1, shape=(1,), dtype=np.float32
        ),
        "pitch_l_angle": spaces.Box(
          low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        ),
        "pitch_r_angle": spaces.Box(
          low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        ),
        "roll_l_angle": spaces.Box(
          low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        ),
        "roll_r_angle": spaces.Box(
          low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
        ),
      }
    )

  @property
  @functools.lru_cache(maxsize=None)
  def action_space(self):
    return SpaceDict(
      {
        "wrist_vel": spaces.Box(
          low=-1, high=1, shape=(1,), dtype=np.float32
        ),
        "pitch_l_vel": spaces.Box(
          low=-1, high=1, shape=(1,), dtype=np.float32
        ),
        "pitch_r_vel": spaces.Box(
          low=-1, high=1, shape=(1,), dtype=np.float32
        ),
        "roll_l_vel": spaces.Box(
          low=-1, high=1, shape=(1,), dtype=np.float32
        ),
        "roll_r_vel": spaces.Box(
          low=-1, high=1, shape=(1,), dtype=np.float32
        ),
      }
    )

  def get_states(self):
    wrist_joint = self.get_joint_states()[self.wrist_joint_id]
    finger_l = self.get_joint_states()[self.gripper_joint_ids[0]]
    finger_r = self.get_joint_states()[self.gripper_joint_ids[1]]
    pitch_l = self.get_joint_states()[self.pitch_joint_ids[0]]
    pitch_r = self.get_joint_states()[self.pitch_joint_ids[1]]
    roll_l = self.get_joint_states()[self.roll_joint_ids[0]]
    roll_r = self.get_joint_states()[self.roll_joint_ids[1]]

    states = self.state_space.new()
    # TODO convert to euler
    states.wrist_angle = wrist_joint.joint_position
    states.finger_l = finger_l.joint_position
    states.finger_r = finger_r.joint_position
    states.pitch_l_angle = pitch_l.joint_position
    states.pitch_r_angle = pitch_r.joint_position
    states.roll_l_angle = roll_l.joint_position
    states.roll_r_angle = roll_r.joint_position
    return states

  def _states_to_joint_position(self, states):
    joint_position = np.zeros(self.num_dofs)
    joint_position[self.wrist_joint_id] = states.wrist_angle
    joint_position[self.gripper_joint_ids[0]] = states.finger_l
    joint_position[self.gripper_joint_ids[1]] = states.finger_r
    joint_position[self.pitch_joint_ids[0]] = states.pitch_l_angle
    joint_position[self.pitch_joint_ids[1]] = states.pitch_r_angle
    joint_position[self.roll_joint_ids[0]] = states.roll_l_angle
    joint_position[self.roll_joint_ids[1]] = states.roll_r_angle
    return joint_position

  def set_actions(self, actions):
    actions = AttrDict(actions)
    states = self.get_states()
    # action is the desired state, overwrite states with actions to get it
    states.wrist_angle += (actions.wrist_vel * self.wrist_vel)
    states.finger_l = 0.00
    states.finger_r = 0.00
    states.pitch_l_angle += (actions.pitch_l_vel * self.pitch_vel)
    states.pitch_r_angle += (actions.pitch_r_vel * self.pitch_vel)
    states.roll_l_angle += (actions.roll_l_vel * self.roll_vel)
    states.roll_r_angle += (actions.roll_r_vel * self.roll_vel)
    joint_position = self._states_to_joint_position(states)
    max_forces = np.ones(self.num_dofs) * self.MAX_FORCES
    max_forces[self.gripper_joint_ids] *= 1
    max_forces[self.pitch_joint_ids] *= 100
    max_forces[self.roll_joint_ids] *= 100
    self.set_joint_position(
      joint_position, max_forces, use_joint_effort_limits=False
    )

  @property
  def digit_links(self):
    return [self.get_joint_index_by_name(name) for name in self.digit_joint_names]

  @property
  @functools.lru_cache(maxsize=None)
  def wrist_joint_id(self):
    return self.free_joint_indices.index(self.get_joint_index_by_name(self.wrist_joint_name))

  @property
  @functools.lru_cache(maxsize=None)
  def gripper_joint_ids(self):
    return [
      self.free_joint_indices.index(self.get_joint_index_by_name(name))
      for name in self.gripper_names
    ]

  @property
  @functools.lru_cache(maxsize=None)
  def pitch_joint_ids(self):
    return [
      self.free_joint_indices.index(self.get_joint_index_by_name(name))
      for name in self.pitch_names
    ]

  @property
  @functools.lru_cache(maxsize=None)
  def roll_joint_ids(self):
    return [
      self.free_joint_indices.index(self.get_joint_index_by_name(name))
      for name in self.roll_names
    ]


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


if __name__ == '__main__':
  arr = char_to_pixels('a')
  print(arr)
