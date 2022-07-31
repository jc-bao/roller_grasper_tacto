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
from scipy.spatial.transform import Rotation as R
import hydra
from attrdict import AttrDict
import os, sys

from utils import RollerGrapser, Camera, convert_obs_to_obs_space, unifrom_sample_quaternion, char_to_pixels, pairwise_registration


class RollerEnv(gym.Env):
  def __init__(self, cfg: AttrDict):
    # configuration
    self.cfg = self.update_params(cfg)
    sys.path.append(hydra.utils.get_original_cwd())

    # physical engine
    self._p = px.init(mode=p.DIRECT)
    self.robot = RollerGrapser(self.cfg.robot_params, self.cfg.init_state)
    self.obj = px.Body(urdf_path=self.cfg.obj_urdf,
                      base_position=[0.000, 0, 0.13], global_scaling=1)
    self.obj_copy = px.Body(urdf_path=self.cfg.obj_urdf,
                          base_position=[0.0, 0.15, 0.13], global_scaling=1, use_fixed_base=True)
    self.obj_target = px.Body(urdf_path=self.cfg.obj_urdf,
                            base_position=[0.0, -0.15, 0.13], global_scaling=1, use_fixed_base=True)
    self.sensor = tacto.Sensor(
      width=self.cfg.sensor_x/2, height=self.cfg.sensor_y/2, visualize_gui=False, config_path='/juno/u/chaoyi/rl/roller_grasper_tacto/assets/sensors/roller.yml')
    self.world_cam = Camera(self.cfg.cam.world_cam)
    self.left_cam = Camera(self.cfg.cam.left_cam)
    self.right_cam = Camera(self.cfg.cam.right_cam)

    # create o3d render window
    self.o3dvis = self.init_o3d_render()
    self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    self.reset()

  def update_params(self, cfg: AttrDict):
    # render params
    cfg.sensor_size = max(cfg.sensor_x, cfg.sensor_y)
    cfg.sensor_x_start = (cfg.sensor_size - cfg.sensor_x)//2
    cfg.sensor_x_end = (cfg.sensor_size + cfg.sensor_x)//2
    cfg.sensor_y_start = (cfg.sensor_size - cfg.sensor_y)//2
    cfg.sensor_y_end = (cfg.sensor_size + cfg.sensor_y)//2
    cfg.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(cfg.sensor_x, cfg.sensor_y,
                                                                     cfg.sensor_size/2, cfg.sensor_size/2, cfg.sensor_x/2, cfg.sensor_y/2)
    for k, cam in cfg.cam.items():
      if cam.width is None:
        cfg['cam'][k]['width'] = cfg.render_x
      if cam.height is None:
        cfg['cam'][k]['height'] = cfg.render_y

    # SLAM params
    cfg.max_correspondence_distance_coarse = cfg.voxel_size * 15
    cfg.max_correspondence_distance_fine = cfg.voxel_size * 1.5
    return cfg

  def init_o3d_render(self):
    o3dvis = o3d.visualization.Visualizer()
    o3dvis.create_window(
      width=self.cfg.render_x, height=self.cfg.render_y, visible=False)
    o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
    o3dvis.get_view_control().set_front(np.array([1, 1, 1]))
    o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
    o3dvis.get_render_option().background_color = (0.5, 0.5, 0.5)
    return o3dvis

  def step(self, action):
    self.steps += 1
    done = self._done()
    reward = int(done)
    info = {}
    self.robot.set_actions(action)

    # update ghost object position
    self.obj_copy.set_base_pose(
      np.asarray(self.obj_copy.init_base_position)-np.asarray(self.obj.init_base_position) + np.asarray(self.obj.get_base_pose()[0]), self.obj.get_base_pose()[1])

    # update object position esitimation
    new_roll = self.robot.get_joint_state_by_name('joint4_left').joint_position
    delta_roll = new_roll - self.old_roll
    self.old_roll = new_roll
    roller_width = self.robot.get_link_state_by_name('joint5_right')[
      0][1] - self.robot.get_link_state_by_name('joint5_left')[0][1] - 0.04
    delta_obj_roll = delta_roll*0.04/roller_width
    self.obj_relative_angle += delta_obj_roll

    # update camera position with the roller
    self.obj_angle = R.from_quat(
      self.obj.get_base_pose()[1]).as_euler('xyz')[0]
    angle_to_vertial_axis = (self.obj_angle+np.pi/4) % (np.pi/2) - np.pi/4
    self.depth_cam_distance = self.cfg.depth_cam_min_distance / \
      abs(np.cos(angle_to_vertial_axis))

    # using object current orientation to record if do the loop closure 
    if abs(self.obj_angle - np.pi) < np.pi/12 and ~self.closed_loop:
      self.cfg.start_detect_loop_idx = len(self.pcd_esitimated)

    p.stepSimulation()

    self.obs = self._get_obs()
    return self.obs, reward, done, info

  def _done(self):
    return (self.steps % 100 == 0)

  def _get_obs(self):
    world_cam_color, world_cam_depth = self.world_cam.get_image()
    left_cam_color, left_cam_depth = self.left_cam.get_image()
    right_cam_color, right_cam_depth = self.right_cam.get_image()
    # update objects positions registered with sensor
    self.sensor.update()
    colors, depths = self.sensor.render()
    obj_pose = self.obj.get_base_pose()
    return AttrDict(
      {
        "world_cam": {"color": world_cam_color, "depth": world_cam_depth},
        "left_cam": {"color": left_cam_color, "depth": left_cam_depth}, 
        "right_cam": {"color": right_cam_color, "depth": right_cam_depth}, 
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
    # Move the object to random location
    dx, dy = np.random.randn(2) * 0.0
    x, y, z = self.obj.init_base_position
    position = [x + dx, y + dy, z]
    # obj_orn = unifrom_sample_quaternion()
    obj_orn = [0, 0, 0, 1]
    self.obj.set_base_pose(position, obj_orn)
    # setup camera
    self.depth_cam_distance = self.cfg.depth_cam_min_distance
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
    sensor_far = self.cfg.sensor_far*1
    proj_matrix = p.computeProjectionMatrixFOV(
      fov=90, aspect=float(self.cfg.sensor_size)/self.cfg.sensor_size,
      nearVal=self.cfg.sensor_near, farVal=sensor_far)
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
      width=self.cfg.sensor_size, height=self.cfg.sensor_size, viewMatrix=view_matrix,
      projectionMatrix=proj_matrix,
      renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgbImg = np.asarray(rgbImg)
    depthImg = np.asarray(depthImg)
    # point cloud data
    rgbImg = Image.fromarray(rgbImg, mode='RGBA').convert('RGB')
    color = o3d.geometry.Image(np.array(rgbImg))
    depth = sensor_far * self.cfg.sensor_near / \
      (sensor_far - (sensor_far - self.cfg.sensor_near) * depthImg)
    depthImg = depth/sensor_far
    depthImg[depthImg > 0.98] = 0

    depth = o3d.geometry.Image((depthImg*255).astype(np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
      color, depth, depth_scale=1/sensor_far, depth_trunc=1000, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd, self.cfg.pinhole_camera_intrinsic)
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
      voxel_size=self.cfg.voxel_size)
    pcd_real.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    self.pcd_goal = pcd_real

    return self.obs

  def render(self, mode="human"):
    if mode == "human":
      color = [digit.color for digit in self.obs.sensor]
      depth = [digit.depth for digit in self.obs.sensor]
      self.sensor.updateGUI(color, depth)
    elif mode == "rgb_array":
      rgb_array = np.ones((2048, 2048, 3)).astype(np.uint8)
      # frame data
      view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.1],
        distance=0.2,
        yaw=90,
        pitch=-20,
        roll=0,
        upAxisIndex=2)
      proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(self.cfg.render_x)/self.cfg.render_y,
        nearVal=0.1, farVal=100.0)
      (_, _, px, _, _) = p.getCameraImage(
        width=self.cfg.render_x, height=self.cfg.render_y, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
      )
      rgb_array[:self.cfg.render_y,
                :self.cfg.render_x, :] = np.array(px)[..., :3]

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
      pcd_real_combined = o3d.geometry.PointCloud()
      pcd_esitimated_combined = o3d.geometry.PointCloud()
      for cam_id in range(1, 3):
        y_start, x_start = self.cfg.sensor_size*cam_id, 0
        # depth data
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
          cameraTargetPosition=self.obj_copy.init_base_position,
          distance=self.depth_cam_distance,
          yaw=180*cam_id,
          pitch=0,
          roll=0,
          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(
          fov=90, aspect=float(self.cfg.sensor_x)/self.cfg.sensor_y,
          nearVal=self.cfg.sensor_near, farVal=self.cfg.sensor_far)
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
          width=self.cfg.sensor_size, height=self.cfg.sensor_size, viewMatrix=view_matrix,
          projectionMatrix=proj_matrix,
          renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgbImg = np.asarray(rgbImg)[
          self.cfg.sensor_y_start:self.cfg.sensor_y_end, self.cfg.sensor_x_start:self.cfg.sensor_x_end]
        depthImg = np.asarray(depthImg)[
          self.cfg.sensor_y_start:self.cfg.sensor_y_end, self.cfg.sensor_x_start:self.cfg.sensor_x_end]
        rgb_array[y_start:y_start+self.cfg.sensor_y, x_start:x_start +
                  self.cfg.sensor_x, :] = np.expand_dims(depthImg*256, 2).repeat(3, axis=2)
        txt = char_to_pixels(f'depth{cam_id}')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        x_start += self.cfg.sensor_x

        # point cloud data
        rgbImg = Image.fromarray(rgbImg, mode='RGBA').convert('RGB')
        color = o3d.geometry.Image(np.array(rgbImg))
        depth = self.cfg.sensor_far * self.cfg.sensor_near / \
          (self.cfg.sensor_far - (self.cfg.sensor_far - self.cfg.sensor_near) * depthImg)
        depthImg = depth/self.cfg.sensor_far
        depthImg[depthImg > 0.98] = 0
        depth = o3d.geometry.Image((depthImg*255).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
          color, depth, depth_scale=1/self.cfg.sensor_far, depth_trunc=1000, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
          rgbd, self.cfg.pinhole_camera_intrinsic)

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

        # object pose relative to camera
        real_obj2worldTrans = np.zeros((4, 4))
        real_obj2worldTrans[:3, :3] = R.from_quat(
          self.obj_copy.get_base_pose()[1]).as_matrix()
        real_obj2worldTrans[:3, 3] = (np.asarray(self.obj_copy.get_base_pose()[
                                      0]) - np.asarray(self.obj_copy.init_base_position))
        real_obj2worldTrans[3, 3] = 1
        real_world2objTrans = np.linalg.inv(real_obj2worldTrans)
        # esitimate object orientation
        esitimated_obj2worldTrans = np.zeros((4, 4))
        obj_init_rot = R.from_quat(self.obj_copy.init_base_orientation)
        local_rot = R.from_euler('x', self.obj_relative_angle)
        obj_rot = local_rot*obj_init_rot
        esitimated_obj2worldTrans[:3, :3] = obj_rot.as_matrix()
        esitimated_obj2worldTrans[:3, 3] = [0, 0, 0]
        esitimated_obj2worldTrans[3, 3] = 1
        esitimated_world2objTrans = np.linalg.inv(esitimated_obj2worldTrans)

        # draw point cloud in camera frame
        self.o3dvis.add_geometry(pcd)
        self.o3dvis.add_geometry(copy.deepcopy(
          self.world_frame).transform(world2camTrans))
        self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
        self.o3dvis.get_view_control().set_front(np.array([0, 0, -1]))
        self.o3dvis.get_view_control().set_up(np.array([0, -1, 0]))
        color = self.o3dvis.capture_screen_float_buffer(do_render=True)
        rgb_array[y_start:y_start+self.cfg.render_x,
                  x_start:x_start+self.cfg.render_y, :] = np.asarray(color)*256
        txt = char_to_pixels(f'PC{cam_id} in cam')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        self.o3dvis.clear_geometries()
        x_start += self.cfg.sensor_size

        # draw point cloud in world frame
        self.o3dvis.add_geometry(self.world_frame)
        pcd.transform(cam2worldTrans)
        self.o3dvis.add_geometry(pcd)
        self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
        self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
        self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
        color = self.o3dvis.capture_screen_float_buffer(do_render=True)
        rgb_array[y_start:y_start+self.cfg.render_x,
                  x_start:x_start+self.cfg.render_y, :] = np.asarray(color)*256
        txt = char_to_pixels(f'PC{cam_id} in world')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        self.o3dvis.clear_geometries()
        x_start += self.cfg.sensor_x

        # draw object in object frame (for reconstruction)
        pcd_real = copy.deepcopy(pcd)
        pcd_esitimated = copy.deepcopy(pcd)
        pcd_real.transform(real_world2objTrans)
        pcd_esitimated.transform(esitimated_world2objTrans)
        pcd_esitimated.paint_uniform_color((1., 1., 0))
        self.o3dvis.add_geometry(copy.deepcopy(
          pcd_real).translate([0, 0, 0.05]))
        self.o3dvis.add_geometry(copy.deepcopy(
          pcd_esitimated).translate([0, 0, -0.05]))
        self.o3dvis.add_geometry(copy.deepcopy(self.world_frame).transform(
          real_world2objTrans).translate([0, 0, 0.05]))
        self.o3dvis.add_geometry(copy.deepcopy(self.world_frame).transform(
          esitimated_world2objTrans).translate([0, 0, -0.05]))
        self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
        self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
        self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
        color = self.o3dvis.capture_screen_float_buffer(do_render=True)
        rgb_array[y_start:y_start+self.cfg.render_x,
                  x_start:x_start+self.cfg.render_y, :] = np.asarray(color)*256
        txt = char_to_pixels(f'PC{cam_id} in obj')
        rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
        self.o3dvis.clear_geometries()
        x_start += self.cfg.sensor_x
        pcd_real_down = pcd_real.voxel_down_sample(voxel_size=self.cfg.voxel_size)
        pcd_real_combined += pcd_real_down
        pcd_esitimated_down = pcd_esitimated.voxel_down_sample(
          voxel_size=self.cfg.voxel_size)
        pcd_esitimated_combined += pcd_esitimated_down
        pcd_combined += pcd.voxel_down_sample(voxel_size=self.cfg.voxel_size)
      pcd_esitimated_combined.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
      pcd_combined.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
      self.pcd_real.append(pcd_real_combined)
      self.pcd_esitimated.append(pcd_esitimated_combined)
      self.pcds.append(pcd_combined)

      # draw ground truth
      y_start, x_start = self.cfg.sensor_size*3, 0
      self.o3dvis.add_geometry(copy.deepcopy(
        self.world_frame).transform(real_world2objTrans))
      for pc in self.pcd_real:
        self.o3dvis.add_geometry(pc)
      self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
      self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
      self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
      color = self.o3dvis.capture_screen_float_buffer(do_render=True)
      rgb_array[y_start:y_start+self.cfg.render_x, x_start:x_start +
                self.cfg.render_y, :] = np.asarray(color)*256
      txt = char_to_pixels(f'Ground truth')
      rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
      self.o3dvis.clear_geometries()
      x_start += self.cfg.sensor_x

      # aling with ICP
      y_start, x_start = self.cfg.sensor_size*3, self.cfg.sensor_size
      self.o3dvis.add_geometry(copy.deepcopy(
        self.world_frame).transform(real_world2objTrans))
      # update pose graph
      pc_num = len(self.pcd_esitimated)
      if pc_num == 1 and self.cfg.use_pcd_goal:
        self.pcd_real.append(self.pcd_goal)
        self.pcd_esitimated.append(self.pcd_goal)
        pc_num += 1
      source_id = pc_num-1
      for i in range(0, min(pc_num-1, self.cfg.matching_num)):
        target_id = pc_num-2-i
        transformation_icp, information_icp, rmse, fitness = pairwise_registration(
          self.pcd_esitimated[source_id], self.pcd_esitimated[target_id],
          self.cfg.max_correspondence_distance_coarse,
          self.cfg.max_correspondence_distance_fine)
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
      if pc_num > self.cfg.start_detect_loop_idx and not self.closed_loop:
        start_id = max(pc_num-self.cfg.start_detect_loop_idx -
                       self.cfg.detect_loop_extend, 0)
        end_id = min(pc_num-self.cfg.start_detect_loop_idx +
                     self.cfg.detect_loop_extend+1, pc_num)
        need_GO = False
        for target_id in range(start_id, end_id):
          transformation_icp, information_icp, rmse, fitness = pairwise_registration(
            self.pcd_esitimated[source_id], self.pcd_esitimated[target_id],
            self.cfg.max_correspondence_distance_coarse,
            self.cfg.max_correspondence_distance_fine)

          if fitness < self.cfg.fit_bar:
            need_GO = True
            self.closed_loop = True
            self.pose_graph.edges.append(
              o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                       target_id,
                                                       transformation_icp,
                                                       information_icp,
                                                       uncertain=True))
        if need_GO:
          option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.cfg.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
          o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
          self.odometry = np.array(self.pose_graph.nodes[-1].pose)
      for point_id, pc in enumerate(self.pcd_esitimated):
        self.o3dvis.add_geometry(copy.deepcopy(pc).transform(
          self.pose_graph.nodes[point_id].pose))
      self.o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
      self.o3dvis.get_view_control().set_front(np.array([1, 0, 0]))
      self.o3dvis.get_view_control().set_up(np.array([0, 0, 1]))
      color = self.o3dvis.capture_screen_float_buffer(do_render=True)
      rgb_array[y_start:y_start+self.cfg.render_y, x_start:x_start +
                self.cfg.render_x, :] = np.asarray(color)*256
      txt = char_to_pixels(f'Estimated')
      rgb_array[y_start:y_start+16, x_start:x_start+64, :] = txt
      txt = char_to_pixels(f'real:{self.obj_angle:.3f}')
      rgb_array[y_start+16:y_start+32, x_start:x_start+64, :] = txt
      txt = char_to_pixels(f'est:{self.obj_relative_angle:.3f}')
      rgb_array[y_start+32:y_start+48, x_start:x_start+64, :] = txt
      # odom = R.from_matrix(self.odometry[:3, :3]).as_euler("xyz")
      # txt = char_to_pixels(f'odx:{odom[0]:.3f}')
      # rgb_array[y_start+48:y_start+64, x_start:x_start+64, :] = txt
      # txt = char_to_pixels(f'ody:{odom[1]:.3f}')
      # rgb_array[y_start+64:y_start+80, x_start:x_start+64, :] = txt
      # txt = char_to_pixels(f'odz:{odom[2]:.3f}')
      # rgb_array[y_start+80:y_start+96, x_start:x_start+64, :] = txt
      odm = np.linalg.norm(R.from_matrix(self.odometry[:3, :3]).as_rotvec())
      txt = char_to_pixels(f'odm:{odm:.3f}')
      rgb_array[y_start+48:y_start+64, x_start:x_start+64, :] = txt
      if self.closed_loop:
        txt = char_to_pixels(f'Closed Loop')
        rgb_array[y_start+112:y_start+128, x_start:x_start+64, :] = txt
      self.o3dvis.clear_geometries()

      # if self.closed_loop:
      #   self.obj_relahtive_angle = odm
      # else:
      self.obj_relative_angle = 0.99*self.obj_relative_angle + 0.01*odm

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

@hydra.main(config_path="config", config_name="main")
def main(cfg):
  cfg = AttrDict(cfg)
  env = RollerEnv(cfg)
  obs = env.reset()
  imgs = []
  for _ in tqdm(range(cfg.total_steps)):
    # act = env.ezpolicy(obs)
    act = env.action_space.new()
    act['wrist_vel'] = 0.
    act['pitch_l_vel'] = 0.
    act['pitch_r_vel'] = 0.
    act['roll_l_vel'] = 1
    act['roll_r_vel'] = 1
    obs, rew, done, info = env.step(act)
    imgs.append(env.render(mode='rgb_array'))
    if done:
      obs = env.reset()
  skvideo.io.vwrite(f'{cfg.file_prefix}.mp4', np.array(imgs))
  imgs = [Image.fromarray(img) for img in imgs]
  imgs[0].save(f"{cfg.file_prefix}.gif", save_all=True,
               append_images=imgs[1:], duration=50, loop=0)
  env.close()


if __name__ == '__main__':
  main()
