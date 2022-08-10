import dill
import copy
import numpy as np
import pybullet as p
import tacto
import pybulletX as px
from attrdict import AttrMap
import gym
from gym.envs.registration import register
from scipy.spatial.transform import Rotation as R
import skvideo.io
from tqdm import tqdm
from PIL import Image
import open3d as o3d
import dcargs

from utils import Camera, convert_obs_to_obs_space, create_o3dvis, unifrom_sample_quaternion, char_to_pixels, pairwise_registration, PyBulletRecorder
from roller2 import RollerGrapser


class RollerEnv(gym.Env):
  reward_per_step = -0.01

  def __init__(self,
               obj_urdf: str = 'assets/objects/curved_cube.urdf'
               ):
    self._p = px.init(mode=p.DIRECT)
    self.recorder = PyBulletRecorder()

    # render parameters
    # self.sensor_size: int = 512
    # self.sensor_x: int = 128
    # self.sensor_y: int = 512
    self.sensor_size: int = 256
    self.sensor_x: int = 256
    self.sensor_y: int = 64

    # ====REMOVE
    self.sensor_x_start: int = (self.sensor_size - self.sensor_x)//2
    self.sensor_x_end: int = (self.sensor_size + self.sensor_x)//2
    self.sensor_y_start: int = (self.sensor_size - self.sensor_y)//2
    self.sensor_y_end: int = (self.sensor_size + self.sensor_y)//2
    # ====REMOVE

    self.sensor_near: float = 0.001
    self.sensor_far: float = 0.08
    self.depth_cam_min_distance: float = 0.03
    self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.sensor_x, self.sensor_y,
                                                                      self.sensor_y/2, 
                                                                      self.sensor_x/2, 
                                                                      self.sensor_x/2, self.sensor_y/2)

    # SLAM params
    self.matching_num: int = 5
    self.voxel_size: float = 0.00002
    self.fit_bar: float = 0.01  # fitness bar to conduct graph optimization
    self.max_correspondence_distance_coarse: float = self.voxel_size * 15
    self.max_correspondence_distance_fine: float = self.voxel_size * 1.5
    self.start_detect_loop_idx: int = 38
    self.detect_loop_extend: int = 15
    self.use_pcd_goal: bool = False

    # create o3d render window
    self.o3dvis = create_o3dvis(render_size=self.sensor_size)
    self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
      size=0.05)

    # robot parameters
    robot_params = {
      'urdf_path': 'assets/robots/roller.urdf',
      'use_fixed_base': True,
      'base_position': [0, 0, 0.3],
    }
    init_state = AttrMap({
      'wrist_angle': 0,
      'finger_l': -0.008,
      'finger_r': +0.008,
      'pitch_l_angle': np.pi/2,
      'pitch_r_angle': np.pi/2,
      'roll_l_angle': 0,
      'roll_r_angle': 0,
    })
    self.robot = RollerGrapser(robot_params, init_state)
    self.recorder.register_object(self.robot.id, '/juno/u/chaoyi/rl/roller_grasper_tacto/examples/assets/robots/roller_blender.urdf')
    self.obj = px.Body(urdf_path=obj_urdf,
                       base_position=[0.000, -0.04, 0.08], global_scaling=1)
    self.recorder.register_object(self.obj.id, '/juno/u/chaoyi/rl/roller_grasper_tacto/examples/assets/objects/usb.urdf')
    self.obj_copy = px.Body(urdf_path=obj_urdf,
                            base_position=[0.0, 0.15, 0.13], global_scaling=1, use_fixed_base=True)
    self.obj_target = px.Body(urdf_path=obj_urdf,
                              base_position=[0.0, -0.15, 0.13], global_scaling=1, use_fixed_base=True)
    self.ghost_obj = px.Body(urdf_path='assets/objects/rounded_cube_ghost.urdf',
                             base_position=[0.000, 0, 0.18], global_scaling=1, use_fixed_base=True)
    self.obj_base = px.Body(urdf_path='assets/objects/usb_base.urdf', base_position=[0.0, 0.0, 0.04], global_scaling=1, use_fixed_base=True)
    self.recorder.register_object(self.obj_base.id, '/juno/u/chaoyi/rl/roller_grasper_tacto/examples/assets/objects/usb_base.urdf')
    # sensor
    self.sensor = tacto.Sensor(
      width=self.sensor_x, height=self.sensor_y, visualize_gui=False, config_path='assets/sensors/roller.yml')
    self.sensor.add_camera(self.robot.id, self.robot.digit_links)
    self.sensor.add_body(self.obj)
    # cam
    self.camera = Camera(width=self.sensor_size*2, height=self.sensor_size*2)
    self.camera_left = Camera(
      width=self.sensor_size, height=self.sensor_size,
      target=self.obj_copy.init_base_position,
      dist=self.depth_cam_min_distance,
      near=self.sensor_near,
      far=self.sensor_far,
      yaw=180, pitch=0, roll=0,
      crop = [0.2, 1]
    )
    self.camera_right = Camera(
      width=self.sensor_size, height=self.sensor_size,
      target=self.obj_copy.init_base_position,
      dist=self.depth_cam_min_distance,
      near=self.sensor_near,
      far=self.sensor_far,
      yaw=0, pitch=0, roll=0,
      crop = [0.2, 1]
    )
    self.reset()
    self.data = {
      'left_cam': list(), 
      'world2leftcam_trans': list(),
      'right_cam': list(),
      'world2rightcam_trans': list(),
      'obj_trans': list(), 
      'real_delta_trans': list(),
      'estimated_delta_trans': list()
    }
    self.mat_data = {
      'depth': list(), 
      'pose': list()
    }

  def step(self, action):
    self.steps += 1
    done = self._done()
    reward = self.reward_per_step + int(done)
    info = {}
    self.robot.set_actions(action)
    self.obj_copy.set_base_pose(
      np.asarray(self.obj_copy.init_base_position)-np.asarray(self.obj.init_base_position) + 
        np.asarray(self.obj.get_base_pose()[0]), self.obj.get_base_pose()[1])

    new_roll = self.robot.get_joint_state_by_name('joint4_left').joint_position
    delta_roll = new_roll - self.old_roll
    self.old_roll = new_roll
    roller_width = self.robot.get_link_state_by_name('joint5_right')[
      0][1] - self.robot.get_link_state_by_name('joint5_left')[0][1] - 0.04
    self.delta_obj_roll = delta_roll*0.04/roller_width
    self.obj_relative_angle += self.delta_obj_roll

    self.obj_angle = R.from_quat(
      self.obj.get_base_pose()[1]).as_euler('xyz')[0]
    angle_to_vertial_axis = (self.obj_angle+np.pi/4) % (np.pi/2) - np.pi/4
    self.depth_cam_distance = self.depth_cam_min_distance / \
      abs(np.cos(angle_to_vertial_axis))
    if abs(self.obj_angle - np.pi) < np.pi/12 and ~self.closed_loop:
      self.start_detect_loop_idx = len(self.pcd_esitimated)

    p.stepSimulation()
    self.obs = self._get_obs()
    return self.obs, reward, done, info

  def _done(self):
    return (self.steps % 100 == 0)

  def _get_obs(self):
    # update camera pos
    self.camera_left.update_dist(self.depth_cam_distance)
    self.camera_right.update_dist(self.depth_cam_distance)
    cam_color, cam_depth = self.camera.get_image()
    cam_color_left, cam_depth_left = self.camera_left.get_image()
    cam_color_right, cam_depth_right = self.camera_right.get_image()
    # update objects positions registered with sensor
    # self.sensor.update()
    colors, depths = self.sensor.render()
    obj_pose = self.obj.get_base_pose()
    return AttrMap(
      {
        "camera": {"color": cam_color, "depth": cam_depth},
        "camera_left": {"color": cam_color_left, "depth": cam_depth_left},
        "camera_right": {"color": cam_color_right, "depth": cam_depth_right},
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
    self.closed_loop = False
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
    # setup camera
    self.depth_cam_distance = self.depth_cam_min_distance
    # get initial observation
    self.obs = self._get_obs()
    self.pcd_esitimated, self.pcd_real, self.pcds = [], [], []
    self.obj_relative_angle = 0
    self.old_roll = self.robot.get_joint_state_by_name(
      'joint5_left').joint_position
    # setup pose graph
    self.pose_graph = o3d.pipelines.registration.PoseGraph()
    self.odometry = np.identity(4)
    self.pose_graph.nodes.append(
      o3d.pipelines.registration.PoseGraphNode(self.odometry))
    p.stepSimulation()

    # setup initial point cloud

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
      cameraTargetPosition=np.asarray(self.obj_target.init_base_position),
      distance=self.depth_cam_distance,
      yaw=180,
      pitch=0,
      roll=0,
      upAxisIndex=2)
    sensor_far = self.sensor_far*1
    proj_matrix = p.computeProjectionMatrixFOV(
      fov=90, aspect=float(self.sensor_size)/self.sensor_size,
      nearVal=self.sensor_near, farVal=sensor_far)
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
      width=self.sensor_size, height=self.sensor_size, viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgbImg = np.asarray(rgbImg)
    depthImg = np.asarray(depthImg)
    # point cloud data
    rgbImg = Image.fromarray(rgbImg, mode='RGBA').convert('RGB')
    color = o3d.geometry.Image(np.array(rgbImg))
    depth = sensor_far * self.sensor_near / \
      (sensor_far - (sensor_far - self.sensor_near) * depthImg)
    depthImg = depth/sensor_far
    depthImg[depthImg > 0.98] = 0

    depth = o3d.geometry.Image((depthImg*255).astype(np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
      color, depth, depth_scale=1/sensor_far, depth_trunc=1000, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd, self.pinhole_camera_intrinsic)
    # transformation information
    cam2worldTrans = np.array([
      [-1, 0, 0, 0],
      [0, 0, -1, self.depth_cam_distance],
      [0, -1, 0, 0],
      [0, 0, 0, 1]])
    world2camTrans = np.linalg.inv(cam2worldTrans)
    # object pose relative to camera
    real_obj2worldTrans = np.zeros((4, 4))
    real_obj2worldTrans[:3, :3] = R.from_quat(
      self.obj_target.get_base_pose()[1]).as_matrix()
    real_obj2worldTrans[:3, 3] = (np.asarray(self.obj_target.get_base_pose()[
                                  0]) - np.asarray(self.obj_target.init_base_position))
    real_obj2worldTrans[3, 3] = 1
    real_world2objTrans = np.linalg.inv(real_obj2worldTrans)
    pcd.transform(cam2worldTrans)
    pcd_real = copy.deepcopy(pcd)
    pcd_real.transform(real_world2objTrans)
    pcd_real = pcd_real.voxel_down_sample(
      voxel_size=self.voxel_size)
    pcd_real.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    self.pcd_goal = pcd_real

    return self.obs

  def ezpolicy(self, obs):
    action = self.action_space.new()
    return action

  def close(self):
    self.o3dvis.destroy_window()
    self.o3dvis.close()

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


def main(obj_urdf: str = 'assets/objects/usb.urdf', file_name: str = 'debug'):
  env = RollerEnv(obj_urdf=obj_urdf)
  obs = env.reset()
  imgs = []
  left_gel_color = []
  right_gel_color = []
  left_gel_depth = []
  right_gel_depth = []
  camera_img = []
  left_cam_img = []
  right_cam_img = []
  process = {
    'move2obj1': 15, 
    'get_obj1': 10,
    'move_back1': 15,
    'roll1': 66,
    'move2ground1': 34,
    'release1': 10,
    'move2obj2': 40,
    'get_obj2': 10,
    'move_back2': 15,
    'roll2': 62,
    'move2ground2': 10,
    'release2': 10,
    'move2obj3': 30,
    'get_obj3': 10,
    'move_back3': 15,
    'roll3': 75,
    'move2ground3': 10,
    'release3': 20,
  }
  # for _ in tqdm(range(n)):
    # act = env.ezpolicy(obs)
  for k, v in process.items():
    for _ in tqdm(range(v)):
      act = env.action_space.new()
      for key in act.keys():
        act[key] = 0
      if k == 'move2obj1':
        # open finger
        act['finger_l'] = -0.02
        act['finger_r'] = 0.02
        # move to object
        robot_pos = np.asarray(env.robot.get_base_pose()[0])
        obj_pos = np.asarray(env.obj.get_base_pose()[0])
        delta_pos = obj_pos - robot_pos + np.asarray([0, 0, 0.19])
        if np.linalg.norm(delta_pos[:2]) > 0.01:
          delta_pos[2] = 0
        robot_pos += np.clip(delta_pos, -0.005, 0.005)
        env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
      elif k == 'get_obj1':
        # close finger
        act['finger_l'] = 0
        act['finger_r'] = 0
      elif k == 'move_back1':
        act['finger_l'] = 0
        act['finger_r'] = 0
        # move back
        robot_pos = np.asarray(env.robot.get_base_pose()[0])
        delta_pos = env.robot.init_base_position - robot_pos
        if delta_pos[2] > 0.01:
          delta_pos[:2] = 0
        robot_pos += np.clip(delta_pos, -0.005, 0.005)
        env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
        env.obj.set_base_pose(robot_pos-np.asarray([0, 0, 0.185]), env.obj.get_base_pose()[1])
      elif k == 'roll1':
        act['roll_l_vel'] = 0.5
        act['roll_r_vel'] = 0.5
      elif k == 'move2ground1':
        # change pitch angle
        robot_joint = obs['robot']
        pitch = robot_joint['pitch_l_angle']
        delta_pitch = np.pi - pitch
        act['pitch_l_vel'] = np.clip(delta_pitch, -1, 1)
        act['pitch_r_vel'] = np.clip(delta_pitch, -1, 1)
        if np.abs(delta_pitch) < 0.3:
          # move to ground
          robot_pos = np.asarray(env.robot.get_base_pose()[0])
          delta_pos = np.asarray([0,0,env.obj.init_base_position[2]+0.192]) - robot_pos
          robot_pos += np.clip(delta_pos, -0.005, 0.005)
          env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])  
          env.obj.set_base_pose(robot_pos-np.asarray([0, 0, 0.185]), env.obj.get_base_pose()[1])
      elif k == 'release1':
        # release
        act['finger_l'] = -0.02
        act['finger_r'] = 0.02 
      elif k == 'move2obj2':
        # open finger
        act['finger_l'] = -0.02
        act['finger_r'] = 0.02
        # move to object
        robot_joint = obs['robot']
        pitch = robot_joint['pitch_l_angle']
        act['pitch_l_vel'] = -np.clip(pitch-np.pi/2, -1, 1)
        act['pitch_r_vel'] = -np.clip(pitch-np.pi/2, -1, 1) 
        if np.abs(pitch) < 0.4:
          robot_pos = np.asarray(env.robot.get_base_pose()[0])
          obj_pos = np.asarray(env.obj.get_base_pose()[0])
          delta_pos = obj_pos - robot_pos + np.asarray([0, 0, 0.185])
          robot_pos += np.clip(delta_pos, -0.005, 0.005)
          env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
        else:
          robot_pos = np.asarray(env.robot.get_base_pose()[0])
          delta_pos = np.asarray([0, 0, env.obj.init_base_position[2]+0.2]) - robot_pos
          robot_pos += np.clip(delta_pos, -0.005, 0.005)
          env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
      elif k == 'get_obj2':
        # close finger
        act['finger_l'] = 0
        act['finger_r'] = 0
      elif k == 'move_back2':
        act['finger_l'] = 0
        act['finger_r'] = 0
        # move back
        robot_pos = np.asarray(env.robot.get_base_pose()[0])
        delta_pos = env.robot.init_base_position - robot_pos
        robot_pos += np.clip(delta_pos, -0.005, 0.005)
        env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
        env.obj.set_base_pose(robot_pos-np.asarray([0, 0, 0.178]), np.asarray([0.707, 0.0,0.707,0.0]))
      elif k == 'roll2':
        act['roll_l_vel'] = 0.5
        act['roll_r_vel'] = 0.5
      elif k == 'move2ground2':
        robot_pos = np.asarray(env.robot.get_base_pose()[0])
        delta_pos = np.asarray([0,0,env.obj.init_base_position[2]+0.19]) - robot_pos
        robot_pos += np.clip(delta_pos, -0.005, 0.005)
        env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])  
        env.obj.set_base_pose(robot_pos-np.asarray([0, 0, 0.190]), np.asarray([0.0,-0.707,0.0,0.707]))
      elif k == 'release2':
        # release
        act['finger_l'] = -0.02
        act['finger_r'] = 0.02
      elif k == 'move2obj3':
        # open finger
        act['finger_l'] = -0.02
        act['finger_r'] = 0.02
        # move to object
        robot_joint = obs['robot']
        pitch = robot_joint['pitch_l_angle']
        act['pitch_l_vel'] = -np.clip(pitch, -1, 1)
        act['pitch_r_vel'] = -np.clip(pitch, -1, 1) 
        if np.abs(pitch) < 0.4:
          robot_pos = np.asarray(env.robot.get_base_pose()[0])
          obj_pos = np.asarray(env.obj.get_base_pose()[0])
          delta_pos = obj_pos - robot_pos + np.asarray([0, 0, 0.185])
          robot_pos += np.clip(delta_pos, -0.005, 0.005)
          env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
        else:
          robot_pos = np.asarray(env.robot.get_base_pose()[0])
          delta_pos = np.asarray([0, 0, env.obj.init_base_position[2]+0.2]) - robot_pos
          robot_pos += np.clip(delta_pos, -0.005, 0.005)
          env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
      elif k == 'get_obj3':
        # close finger
        act['finger_l'] = 0
        act['finger_r'] = 0
      elif k == 'move_back3':
        act['finger_l'] = 0
        act['finger_r'] = 0
        # move back
        robot_pos = np.asarray(env.robot.get_base_pose()[0])
        delta_pos = env.robot.init_base_position - robot_pos
        robot_pos += np.clip(delta_pos, -0.005, 0.005)
        env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])
        env.obj.set_base_pose(robot_pos-np.asarray([0, 0, 0.178]), np.asarray([0, -0.707,0,0.707]))
      elif k == 'roll3':
        act['roll_l_vel'] = 0.5
        act['roll_r_vel'] = 0.5
      elif k == 'move2ground3':
        robot_pos = np.asarray(env.robot.get_base_pose()[0])
        delta_pos = np.asarray([0,0,env.obj.init_base_position[2]+0.19]) - robot_pos
        robot_pos += np.clip(delta_pos, -0.005, 0.005)
        env.robot.set_base_pose(robot_pos, env.robot.get_base_pose()[1])  
        env.obj.set_base_pose(robot_pos-np.asarray([0, 0, 0.190]), np.asarray([0.707,0,-0.707,0]))
      elif k == 'release3':
        # release
        act['roll_l_vel'] = 1
        act['roll_r_vel'] = -1
      
      obs, rew, done, info = env.step(act)
      env.recorder.add_keyframe()

      # rendering items
      camera_img.append(obs.camera.color)
      left_cam_img.append(obs.camera_left.color)
      right_cam_img.append(obs.camera_right.color)
      left_gel_color.append(obs.sensor[0].color)
      left_gel_depth.append(np.expand_dims(obs.sensor[0].depth*256, 2).repeat(3, axis=2).astype(np.uint8))
      right_gel_color.append(obs.sensor[1].color)
      right_gel_depth.append(np.expand_dims(obs.sensor[1].depth*256, 2).repeat(3, axis=2).astype(np.uint8))

  for k, img in enumerate([camera_img, left_cam_img, right_cam_img, left_gel_color, left_gel_depth, right_gel_color, right_gel_depth]):
    imgs = [Image.fromarray(i) for i in img]
    imgs[0].save(f"render/{file_name}_{k}.gif", save_all=True,
                append_images=imgs[1:])
  env.recorder.save('render/blender.pkl')
  env.close()


if __name__ == '__main__':
  dcargs.cli(main)
