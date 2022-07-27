import copy
import enum
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
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from utils import Camera, convert_obs_to_obs_space, unifrom_sample_quaternion, char_to_pixels, pairwise_registration

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
    self.sensor_near, self.sensor_far = 0.01, 0.08
    self.depth_cam_distance = 0.05 
    self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.sensor_x, self.sensor_y, max(
      self.sensor_y, self.sensor_x)/2, max(self.sensor_y, self.sensor_x)/2, self.sensor_x/2, self.sensor_y/2)

    # SLAM params
    self.matching_num = 5
    self.voxel_size = 0.0001
    self.fit_bar = 0.01 # fitness bar to conduct graph optimization
    self.max_correspondence_distance_coarse = self.voxel_size * 15
    self.max_correspondence_distance_fine = self.voxel_size * 1.5

    # create o3d render window
    self.o3dvis = o3d.visualization.Visualizer()
    self.o3dvis.create_window(
      width=self.sensor_x, height=self.sensor_y, visible=False)
    self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
    self.o3dvis.get_view_control().set_front(np.array([1, 1, 1]))
    self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))

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
      'pitch_l_angle': 0,
      'pitch_r_angle': 0,
      'roll_l_angle': 0,
      'roll_r_angle': 0,
    })
    self.robot = RollerGrapser(robot_params, init_state)
    self.obj = px.Body(urdf_path='assets/objects/curved_cube.urdf',
                       base_position=[0.000, 0, 0.13], global_scaling=1)
    self.obj_copy = px.Body(urdf_path='assets/objects/curved_cube.urdf',
                            base_position=[0.0, 0.2, 0.13], global_scaling=1, use_fixed_base=True)
    self.ghost_obj = px.Body(urdf_path='assets/objects/rounded_cube_ghost.urdf',
                             base_position=[0.000, 0, 0.18], global_scaling=1, use_fixed_base=True)
    self.sensor = tacto.Sensor(
      width=self.sensor_x/2, height=self.sensor_y/2, visualize_gui=False, config_path='assets/sensors/roller.yml')
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
    self.obj_copy.set_base_pose(
      np.asarray(self.obj_copy.init_base_position)-np.asarray(self.obj.init_base_position )+ np.asarray(self.obj.get_base_pose()[0]), self.obj.get_base_pose()[1])
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
    # get initial observation
    self.obs = self._get_obs()
    self.pcd_right, self.pcd_left, self.pcds= [], [], []
    # setup pose graph
    self.pose_graph = o3d.pipelines.registration.PoseGraph()
    self.odometry = np.identity(4)
    self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(self.odometry))
    return self.obs

  def render(self, mode="human"):
    if mode == "human":
      color = [digit.color for digit in self.obs.sensor]
      depth = [digit.depth for digit in self.obs.sensor]
      self.sensor.updateGUI(color, depth)
    elif mode == "rgb_array":
      rgb_array = np.ones((1024, 1024, 3), dtype=np.uint8)

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
      rgb_array[:self.window_y, :self.window_x,:] = np.array(px)[..., :3]

      # sensor data
      for i in range(len(self.obs.sensor)):
        color = self.obs.sensor[i].color
        depth = self.obs.sensor[i].depth
        shape_x, shape_y = depth.shape[:2]
        rgb_array[shape_x*i:shape_x*(i+1), :shape_y, :] = color
        rgb_array[shape_x*i:shape_x*(i+1), shape_y:shape_y*2,
                  :] = np.expand_dims(depth*256, 2).repeat(3, axis=2)
        txt = char_to_pixels(f'GEL{i+1}')
        rgb_array[shape_x*i:shape_x*i+16, :64, :] = txt
        
      pcd_combined = o3d.geometry.PointCloud()
      for cam_id in range(1,3):
        y_start, x_start = self.sensor_y*cam_id, 0
        # depth data
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
          cameraTargetPosition=self.obj_copy.init_base_position,
          distance=self.depth_cam_distance,
          yaw=180*cam_id,
          pitch=0,
          roll=0,
          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
          fov=90, aspect=float(self.sensor_x)/self.sensor_y,
          nearVal=self.sensor_near, farVal=self.sensor_far)
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
          width=self.sensor_x, height=self.sensor_y, viewMatrix=view_matrix,
          projectionMatrix=proj_matrix,
          renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array[y_start:y_start+self.sensor_y, x_start:x_start+self.sensor_x,:] = np.expand_dims(depthImg*256, 2).repeat(3, axis=2)
        txt = char_to_pixels(f'depth{cam_id}')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        x_start += self.sensor_x

        # point cloud data
        rgbImg = Image.fromarray(rgbImg, mode='RGBA').convert('RGB')
        color = o3d.geometry.Image(np.array(rgbImg))
        depth = self.sensor_far * self.sensor_near / \
          (self.sensor_far - (self.sensor_far - self.sensor_near) * depthImg)
        depthImg = depth/self.sensor_far
        depthImg[depthImg > 0.98] = 0
        depth = o3d.geometry.Image((depthImg*255).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
          color, depth, depth_scale=1/self.sensor_far, depth_trunc=1000, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
          rgbd, self.pinhole_camera_intrinsic)

        # transformation information
        if cam_id == 2:
          cam2worldTrans = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, -self.depth_cam_distance],
            [0, -1, 0, 0],
            [0, 0, 0, 1]])
        elif cam_id == 1:
          cam2worldTrans = np.array([
            [-1, 0, 0, 0],
            [0, 0, -1, self.depth_cam_distance],
            [0, -1, 0, 0],
            [0, 0, 0, 1]])
        else:
          raise NotImplementedError
        world2camTrans = np.linalg.inv(cam2worldTrans)
        obj2worldTrans = np.zeros((4,4))
        obj2worldTrans[:3,:3] = R.from_quat(self.obj_copy.get_base_pose()[1]).as_matrix()
        obj2worldTrans[:3,3] = (np.asarray(self.obj_copy.get_base_pose()[0]) - np.asarray(self.obj_copy.init_base_position))
        obj2worldTrans[3,3] = 1
        world2objTrans = np.linalg.inv(obj2worldTrans)

        # draw point cloud in camera frame
        self.o3dvis.add_geometry(pcd)
        self.o3dvis.add_geometry(copy.deepcopy(self.world_frame).transform(world2camTrans))
        self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
        self.o3dvis.get_view_control().set_front(np.array([0, 0, -1]))
        self.o3dvis.get_view_control().set_up(np.array([0, -1, 0]))
        color = self.o3dvis.capture_screen_float_buffer(do_render=True)
        rgb_array[y_start:y_start+self.sensor_y, x_start:x_start+self.sensor_x,:] = np.asarray(color)*256
        txt = char_to_pixels(f'PC{cam_id} in cam')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        self.o3dvis.clear_geometries()
        x_start += self.sensor_x

        # draw point cloud in world frame
        self.o3dvis.add_geometry(self.world_frame)
        pcd.transform(cam2worldTrans)
        self.o3dvis.add_geometry(pcd)
        self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
        self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
        self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
        color = self.o3dvis.capture_screen_float_buffer(do_render=True)
        rgb_array[y_start:y_start+self.sensor_y, x_start:x_start+self.sensor_x,:] = np.asarray(color)*256
        txt = char_to_pixels(f'PC{cam_id} in world')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        self.o3dvis.clear_geometries()
        x_start += self.sensor_x

        # draw object in object frame (for reconstruction)
        pcd.transform(world2objTrans)
        self.o3dvis.add_geometry(pcd)
        self.o3dvis.add_geometry(copy.deepcopy(self.world_frame).transform(world2objTrans))
        self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
        self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
        self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
        color = self.o3dvis.capture_screen_float_buffer(do_render=True)
        rgb_array[y_start:y_start+self.sensor_y, x_start:x_start+self.sensor_x,:] = np.asarray(color)*256
        txt = char_to_pixels(f'PC{cam_id} in obj')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        self.o3dvis.clear_geometries()
        x_start += self.sensor_x
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if cam_id == 1:
          self.pcd_right.append(pcd_down)
        elif cam_id == 2:
          self.pcd_left.append(pcd_down)
        pcd_combined += pcd_down
      pcd_combined.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
      self.pcds.append(pcd_combined)

      # draw all frames
      y_start, x_start = self.sensor_y*3, 0
      self.o3dvis.add_geometry(copy.deepcopy(self.world_frame).transform(world2objTrans))
      for pc in self.pcds:
        self.o3dvis.add_geometry(pc)
      self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
      self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
      self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
      color = self.o3dvis.capture_screen_float_buffer(do_render=True)
      rgb_array[y_start:y_start+self.sensor_y, x_start:x_start+self.sensor_x,:] = np.asarray(color)*256
      txt = char_to_pixels(f'Unaligned PCs')
      rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
      self.o3dvis.clear_geometries()
      x_start += self.sensor_x

      # aling with ICP
      y_start, x_start = self.sensor_y*3, self.sensor_x
      self.o3dvis.add_geometry(copy.deepcopy(self.world_frame).transform(world2objTrans))
      # update pose graph
      pc_num = len(self.pcds)
      source_id = pc_num-1
      for i in range(0, min(pc_num-1, self.matching_num)):
        target_id = pc_num-2-i
        transformation_icp, information_icp, rmse, fitness = pairwise_registration(
          self.pcds[source_id], self.pcds[target_id],
          self.max_correspondence_distance_coarse,
          self.max_correspondence_distance_fine)
        if i == 0:  # odometry case
          self.odometry = np.dot(transformation_icp, self.odometry)
          self.pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
              np.linalg.inv(self.odometry)))
          self.pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=False))
        else:  # loop closure case
          self.pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    transformation_icp,
                                                    information_icp,
                                                    uncertain=True))
      transformation_icp, information_icp, rmse, fitness = pairwise_registration(
        self.pcds[source_id], self.pcds[0],
        self.max_correspondence_distance_coarse,
        self.max_correspondence_distance_fine)

      if fitness < self.fit_bar:
        self.closed_loop = True
        self.pose_graph.edges.append(
          o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                  0,
                                                  transformation_icp,
                                                  information_icp,
                                                  uncertain=True))
        option = o3d.pipelines.registration.GlobalOptimizationOption(
          max_correspondence_distance=self.max_correspondence_distance_fine,
          edge_prune_threshold=0.25,
          reference_node=0)
        o3d.pipelines.registration.global_optimization(
          self.pose_graph,
          o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
          o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
          option)
      
      for point_id, pc in enumerate(self.pcds):
        self.o3dvis.add_geometry(copy.deepcopy(pc).transform(self.pose_graph.nodes[point_id].pose))
      self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
      self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
      self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
      color = self.o3dvis.capture_screen_float_buffer(do_render=True)
      rgb_array[y_start:y_start+self.sensor_y, x_start:x_start+self.sensor_x,:] = np.asarray(color)*256
      txt = char_to_pixels(f'Aligned PCs')
      rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
      txt = char_to_pixels(f'rms0:{rmse:.3f}')
      rgb_array[y_start+16:y_start+32, x_start:x_start+64, :] = txt
      txt = char_to_pixels(f'fit0:{fitness:.3f}')
      rgb_array[y_start+32:y_start+48, x_start:x_start+64, :] = txt
      if self.closed_loop:
        txt = char_to_pixels(f'Closed Loop')
        rgb_array[y_start+48:y_start+64, x_start:x_start+64, :] = txt
      self.o3dvis.clear_geometries()

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

if __name__ == '__main__':
  env = RollerEnv()
  obs = env.reset()
  imgs = []
  for _ in tqdm(range(20)):
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
  env.close()
