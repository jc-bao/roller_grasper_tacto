import pybulletX as px
import collections
import numpy as np
import gym
from PIL import Image, ImageFont, ImageDraw
import open3d as o3d
from typing import List
import pybullet as p
import PySimpleGUI as sg
import pickle
from os import getcwd
from urdfpy import URDF
from os.path import abspath, dirname, basename, splitext
from transforms3d.affines import decompose
from transforms3d.quaternions import mat2quat


class Camera:
  def __init__(self, width: int = 320, height: int = 240, target: List[float] = [0., 0., 0.1], dist: float = 0.1, yaw: float = 90.0, pitch: float = -20.0, roll: float = 0, fov: float = 90.0, near: float = 0.001, far: float = 100.0, crop: List[float] = [1, 1]):
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

  def update_dist(self, dist: float):
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

    rgb = np.asarray(img_arr[2], dtype=np.uint8)[
      self.y_start:self.y_end, self.x_start:self.x_end]  # color image RGB H x W x 3 (uint8)
    dep = np.asarray(img_arr[3], dtype=np.uint8)[
      self.y_start:self.y_end, self.x_start:self.x_end]  # depth image H x W (float32)
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


class PyBulletRecorder:
  class LinkTracker:
    def __init__(self,
                 name,
                 body_id,
                 link_id,
                 link_origin,
                 mesh_path,
                 mesh_scale):
      self.body_id = body_id
      self.link_id = link_id
      self.mesh_path = mesh_path
      self.mesh_scale = mesh_scale
      decomposed_origin = decompose(link_origin)
      orn = mat2quat(decomposed_origin[1])
      orn = [orn[1], orn[2], orn[3], orn[0]]
      self.link_pose = [decomposed_origin[0],
                        orn]
      self.name = name

    def transform(self, position, orientation):
      return p.multiplyTransforms(
        position, orientation,
        self.link_pose[0], self.link_pose[1],
      )

    def get_keyframe(self):
      if self.link_id == -1:
        position, orientation = p.getBasePositionAndOrientation(
          self.body_id)
        position, orientation = self.transform(
          position=position, orientation=orientation)
      else:
        link_state = p.getLinkState(self.body_id,
                                    self.link_id,
                                    computeForwardKinematics=True)
        position, orientation = self.transform(
          position=link_state[4],
          orientation=link_state[5])
      return {
        'position': list(position),
        'orientation': list(orientation)
      }

  def __init__(self):
    self.states = []
    self.links = []

  def register_object(self, body_id, urdf_path, global_scaling=1):
    link_id_map = dict()
    n = p.getNumJoints(body_id)
    link_id_map[p.getBodyInfo(body_id)[0].decode('gb2312')] = -1
    for link_id in range(0, n):
      link_id_map[p.getJointInfo(body_id, link_id)[
        12].decode('gb2312')] = link_id

    dir_path = dirname(abspath(urdf_path))
    file_name = splitext(basename(urdf_path))[0]
    robot = URDF.load(urdf_path)
    for link in robot.links:
      link_id = link_id_map[link.name]
      if len(link.visuals) > 0:
        for i, link_visual in enumerate(link.visuals):
          mesh_scale = [global_scaling,
                        global_scaling, global_scaling]\
            if link_visual.geometry.mesh.scale is None \
            else link_visual.geometry.mesh.scale * global_scaling
          self.links.append(
            PyBulletRecorder.LinkTracker(
              name=file_name + f'_{body_id}_{link.name}_{i}',
              body_id=body_id,
              link_id=link_id,
              link_origin=# If link_id == -1 then is base link,
              # PyBullet will return
              # inertial_origin @ visual_origin,
              # so need to undo that transform
              (np.linalg.inv(link.inertial.origin)
               if link_id == -1
               else np.identity(4)) @
              link_visual.origin * global_scaling,
              mesh_path=dir_path + '/' +
              link_visual.geometry.mesh.filename,
              mesh_scale=mesh_scale))

  def add_keyframe(self):
    # Ideally, call every p.stepSimulation()
    current_state = {}
    for link in self.links:
      current_state[link.name] = link.get_keyframe()
    self.states.append(current_state)

  def prompt_save(self):
    layout = [[sg.Text('Do you want to save previous episode?')],
              [sg.Button('Yes'), sg.Button('No')]]
    window = sg.Window('PyBullet Recorder', layout)
    save = False
    while True:
      event, values = window.read()
      if event in (None, 'No'):
        break
      elif event == 'Yes':
        save = True
        break
    window.close()
    if save:
      layout = [[sg.Text('Where do you want to save it?')],
                [sg.Text('Path'), sg.InputText(getcwd())],
                [sg.Button('OK')]]
      window = sg.Window('PyBullet Recorder', layout)
      event, values = window.read()
      window.close()
      self.save(values[0])
    self.reset()

  def reset(self):
    self.states = []

  def get_formatted_output(self):
    retval = {}
    for link in self.links:
      retval[link.name] = {
        'type': 'mesh',
        'mesh_path': link.mesh_path,
        'mesh_scale': link.mesh_scale,
        'frames': [state[link.name] for state in self.states]
      }
    return retval

  def save(self, path):
    if path is None:
      print("[Recorder] Path is None.. not saving")
    else:
      print("[Recorder] Saving state to {}".format(path))
      pickle.dump(self.get_formatted_output(), open(path, 'wb'))


if __name__ == '__main__':
  PyBulletRecorder()