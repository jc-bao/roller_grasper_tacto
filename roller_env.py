import copy
import logging
import functools
import numpy as np
import pybullet as p
import tacto
import pybulletX as px
from pybulletX.utils.space_dict import SpaceDict
from attrdict import AttrMap
import gym
from gym import spaces
from gym.envs.registration import register
from scipy.spatial.transform import Rotation as R
import skvideo.io
from tqdm import tqdm
from PIL import Image

from utils import Camera, convert_obs_to_obs_space, unifrom_sample_quaternion

log = logging.getLogger(__name__)


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
    actions = AttrMap(actions)
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


class RollerEnv(gym.Env):
  reward_per_step = -0.01

  def __init__(self):
    self._p = px.init(mode=p.DIRECT)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    # p.resetDebugVisualizerCamera(
    #   cameraDistance=0.2, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.0, 0, 0.1])
    # p.setRealTimeSimulation(False)
    # render parameters
    self.window_x, self.window_y = 1024, 512
    self.sensor_x, self.sensor_y = 256, 256
    robot_params = {
      'urdf_path': 'assets/robots/roller.urdf',
      'use_fixed_base': True,
      'base_position': [0, 0, 0.3],
    }
    init_state = AttrMap({
      'wrist_angle': 0,
      'finger_l': -0.008,
      'finger_r': +0.008,
      'pitch_l_angle': 0,
      'pitch_r_angle': 0,
      'roll_l_angle': 0,
      'roll_r_angle': 0,
    })
    self.robot = RollerGrapser(robot_params, init_state)
    self.obj = px.Body(urdf_path='assets/objects/curved_cube.urdf',
                       base_position=[0.000, 0, 0.13], global_scaling=1)
    self.obj_copy = px.Body(urdf_path='assets/objects/curved_cube.urdf',
                       base_position=[0.0, 0.2, 0.13], global_scaling=1, use_fixed_base=
                       True)
    self.ghost_obj = px.Body(urdf_path='assets/objects/rounded_cube_ghost.urdf',
                             base_position=[0.000, 0, 0.18], global_scaling=1, use_fixed_base=True)
    self.sensor = tacto.Sensor(
      width=self.sensor_x, height=self.sensor_y, visualize_gui=False, config_path='assets/sensors/roller.yml')
    self.camera = Camera()
    self.viewer = None
    self.sensor.add_camera(self.robot.id, self.robot.digit_links)
    self.sensor.add_body(self.obj)
    self.reset()

  def step(self, action):
    self.steps += 1
    done = self._done()
    reward = self.reward_per_step + int(done)
    info = {}
    self.robot.set_actions(action)
    self.obj_copy.set_base_pose(self.obj_copy.init_base_position, self.obj.get_base_pose()[1])
    p.stepSimulation()
    self.obs = self._get_obs()
    return self.obs, reward, done, info

  def _done(self):
    return (self.steps % 100 == 0)

  def _get_obs(self):
    cam_color, cam_depth = self.camera.get_image()
    # update objects positions registered with sensor
    self.sensor.update()
    colors, depths = self.sensor.render()
    obj_pose = self.obj.get_base_pose()
    return AttrMap(
      {
        "camera": {"color": cam_color, "depth": cam_depth},
        "sensor": [
          {"color": color, "depth": depth}
          for color, depth in zip(colors, depths)
        ],
        "robot": self.robot.get_states(),
        "object": {
          "position": np.array(obj_pose[0]),
          "orientation": np.array(obj_pose[1]),
        },
        "goal": self.goal
      }
    )

  def reset(self):
    self.steps = 0
    self.robot.reset()
    # sample goal
    self.goal = unifrom_sample_quaternion()
    self.ghost_obj.set_base_pose(self.ghost_obj.init_base_position, self.goal)
    # Move the object to random location
    dx, dy = np.random.randn(2) * 0.0
    x, y, z = self.obj.init_base_position
    position = [x + dx, y + dy, z]
    # obj_orn = unifrom_sample_quaternion()
    obj_orn = [0, 0, 0, 1]
    self.obj.set_base_pose(position, obj_orn)
    # get initial observation
    self.obs = self._get_obs()
    return self.obs

  def render(self, mode="human"):
    if mode == "human":
      color = [digit.color for digit in self.obs.sensor]
      depth = [digit.depth for digit in self.obs.sensor]
      self.sensor.updateGUI(color, depth)
    elif mode == "rgb_array":
      # frame data
      self.window_x = 1024
      self.window_y = 512
      view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.1],
        distance=0.2,
        yaw=90,
        pitch=-20,
        roll=0,
        upAxisIndex=2)
      proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(self.window_x)/self.window_y,
        nearVal=0.1, farVal=100.0)
      (_, _, px, _, _) = p.getCameraImage(
        width=self.window_x, height=self.window_y, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
      )
      rgb_array = np.array(px)
      rgb_array = rgb_array[:, :, :3]
      # sensor data
      for i in range(len(self.obs.sensor)):
        color = self.obs.sensor[i].color
        depth = self.obs.sensor[i].depth
        shape_x, shape_y = depth.shape[:2]
        rgb_array[shape_x*i:shape_x*(i+1), :shape_y, :] = color
        rgb_array[shape_x*i:shape_x*(i+1), shape_y:shape_y*2, :] = np.expand_dims(depth*256, 2).repeat(3, axis=2)
      # depth data
      view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=self.obj_copy.init_base_position,
        distance=0.05,
        yaw=180,
        pitch=0,
        roll=0,
        upAxisIndex=2)
      proj_matrix = p.computeProjectionMatrixFOV(
        fov=90, aspect=float(self.sensor_x)/self.sensor_y,
        nearVal=0.01, farVal=0.08)
      width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=self.sensor_x, height=self.sensor_y, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
      )
      rgb_array[:self.sensor_y, -self.sensor_x:, :] = np.expand_dims(depthImg*256, 2).repeat(3, axis=2)
      return rgb_array

  def ezpolicy(self, obs):
    # ===parse observation===
    goal_orn = R.from_quat(obs['goal'])
    obj_orn = R.from_quat(obs['object']['orientation'])
    robot_joint = obs['robot']
    robot_orn = R.from_euler('z', robot_joint['wrist_angle'])
    pitch = robot_joint['pitch_l_angle'] % (2 * np.pi)
    obj_pos = obs['object']['position']
    obj_pos[..., 2] -= 0.13
    # ===calculate angular velocity===
    diff_orn = goal_orn * obj_orn.inv()
    omega = 0.5 * 2 * ((diff_orn).as_quat())[..., :3]
    omega *= ((diff_orn.as_quat()[..., 3] > 0) * 2 - 1)
    local_omega = robot_orn.apply(omega)
    print(diff_orn.as_quat())
    # local_omega_norm = np.linalg.norm(local_omega)
    # if local_omega_norm > 2:
    #   local_omega /= local_omega_norm
    # ===calculate action===
    err = 0.2
    delta = abs(abs(pitch-np.pi) - np.pi/2)
    action = env.action_space.new()
    if abs(local_omega[1]) < 0.05 and delta < err:
      action['wrist_vel'] = np.clip(local_omega[..., 2], -1, 1)
    else:
      action['wrist_vel'] = np.clip(local_omega[..., 2] -
                                    local_omega[..., 1] * np.tan(pitch), -1, 1)
    if delta < err:
      action['wrist_vel'] *= delta/err
    action['pitch_l_vel'] = np.clip(local_omega[..., 1], -1, 1)
    action['pitch_r_vel'] = np.clip(local_omega[..., 1], -1, 1)
    action['roll_l_vel'] = np.clip(local_omega[..., 0] / np.cos(pitch), -1, 1)
    action['roll_r_vel'] = np.clip(local_omega[..., 0] / np.cos(pitch), -1, 1)
    # compensate for the drop down
    roller_orn_local = R.from_euler('x', pitch)
    roller_orn = roller_orn_local * robot_orn.inv()
    obj_pos_local = roller_orn.apply(obj_pos)
    action['roll_l_vel'] += np.clip(obj_pos_local[..., 2] * 20, -1, 1)
    action['roll_r_vel'] -= np.clip(obj_pos_local[..., 2] * 20, -1, 1)
    return action

  def close(self):
    pass

  def seed(self, seed=None):
    np.random.seed(seed)

  @property
  def observation_space(self):
    return px.utils.SpaceDict(
      {
        "camera": convert_obs_to_obs_space(self.obs.camera),
        "sensor": convert_obs_to_obs_space(self.obs.sensor),
        "robot": self.robot.state_space,
        "object": {
          "position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
          "orientation": gym.spaces.Box(low=-1.0, high=+1.0, shape=(4,)),
        },
      }
    )

  @property
  def action_space(self):
    return copy.deepcopy(self.robot.action_space)


register(
  id="roller-v0", entry_point="roller_env:RollerEnv",
)

if __name__ == '__main__':
  env = RollerEnv()
  obs = env.reset()
  imgs = []
  for _ in tqdm(range(10)):
    # act = env.ezpolicy(obs)
    act = env.action_space.new()
    act['wrist_vel'] = 0.
    act['pitch_l_vel'] = 0.
    act['pitch_r_vel'] = 0.
    act['roll_l_vel'] = 1.
    act['roll_r_vel'] = 1.
    obs, rew, done, info = env.step(act)
    imgs.append(env.render(mode='rgb_array'))
    if done:
      obs = env.reset()
  skvideo.io.vwrite('render/render.mp4', np.array(imgs))
  imgs = [Image.fromarray(img) for img in imgs]
  imgs[0].save("render/render.gif", save_all=True,
               append_images=imgs[1:], duration=50, loop=0)