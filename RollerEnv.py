import time

import cv2
import deepdish as dd
import numpy as np
import pybullet as pb
import pybullet_data
import tacto
import gym


class Camera:
  def __init__(self, cameraResolution=[320, 240]):
    self.cameraResolution = cameraResolution
    camTargetPos = [-0.01, 0, 0.04]
    camDistance = 0.05
    upAxisIndex = 2
    yaw = 0
    pitch = -20.0
    roll = 0
    fov = 60
    nearPlane = 0.01
    farPlane = 100
    self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
      camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
    )
    aspect = cameraResolution[0] / cameraResolution[1]
    self.projectionMatrix = pb.computeProjectionMatrixFOV(
      fov, aspect, nearPlane, farPlane
    )

  def get_image(self):
    img_arr = pb.getCameraImage(
      self.cameraResolution[0],
      self.cameraResolution[1],
      self.viewMatrix,
      self.projectionMatrix,
      shadow=1,
      lightDirection=[1, 1, 1],
      renderer=pb.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = img_arr[2]  # color data RGB
    dep = img_arr[3]  # depth data
    return rgb, dep

def draw_circle(img, state):
  if state is None:
    return img
  # Center coordinates
  center_coordinates = (
    int(state[1] * img.shape[1]), int(state[0] * img.shape[0]))
  # Radius of circle
  radius = 7
  # Red color in BGR
  color = (255, 255, 255)
  # Line thickness of -1 px
  thickness = -1
  # Draw a circle of red color of thickness -1 px
  img = cv2.circle(img, center_coordinates, radius, color, thickness)
  return img


class RollingEnv(gym.Env):
  def __init__(
      self,
      tactoResolution=[120, 160],
      visPyBullet=True,
      visTacto=True,
      recordLogs=False,
      skipFrame=1,
  ):
    """
    Initialize

    Args:
        tactoResolution: tactile output resolution, Default: [120, 160]
        visPyBullet: whether display pybullet, Default: True
        visTacto: whether display tacto GUI, Default: True
        skipFrame: execute the same action for skipFrame+1 frames.
                   Save time to perform longer horizon
    """
    self.tactoResolution = tactoResolution
    self.visPyBullet = visPyBullet
    self.visTacto = visTacto
    self.skipFrame = skipFrame
    # basic parameters
    self.error = 0
    self.grasp_z = 0.055
    self.create_scene()
    self.cam = Camera(cameraResolution=[320, 240])
    self.logs = {"touch": [], "vision": [], "states": [], "goal": None}
    self.recordLogs = recordLogs

  def create_scene(self):
    """
    Create scene and tacto simulator
    """
    # Initialize roller
    roller = tacto.Sensor(
      width=self.tactoResolution[0],
      height=self.tactoResolution[1],
      visualize_gui=self.visTacto,
      config_path='assets/sensors/roller.yml',
    )
    if self.visPyBullet:
      self.physicsClient = pb.connect(pb.GUI)
    else:
      self.physicsClient = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth
    # Set camera view
    pb.resetDebugVisualizerCamera(
      cameraDistance=0.12,
      cameraYaw=0,
      cameraPitch=-20,
      cameraTargetPosition=[0, 0, 0.02],
    )
    pb.loadURDF("plane.urdf")  # Create plane
    rollerURDF = "assets/sensors/roller.urdf"
    # Set upper roller
    rollerPos1 = [0, 0, 0.011]
    rollerOrn1 = pb.getQuaternionFromEuler([0, -np.pi / 2, 0])
    rollerID1 = pb.loadURDF(
      rollerURDF,
      basePosition=rollerPos1,
      baseOrientation=rollerOrn1,
      useFixedBase=True,
    )
    roller.add_camera(rollerID1, [-1])

    # Set lower roller
    rollerPos2 = [0, 0, 0.07]
    rollerOrn2 = pb.getQuaternionFromEuler([0, np.pi / 2, np.pi])
    rollerID2 = pb.loadURDF(
      rollerURDF, basePosition=rollerPos2, baseOrientation=rollerOrn2,
    )
    roller.add_camera(rollerID2, [-1])

    # Create object and GUI controls
    init_xyz = np.array([0, 0.0, 8])

    # Add object to pybullet and tacto simulator
    urdfObj = "assets/objects/sphere_small.urdf"
    objPos = np.array([-1.5, 0, 4]) / 100
    objOrn = pb.getQuaternionFromEuler([0, 0, 0])
    globalScaling = 0.15

    # Add ball urdf into pybullet and tacto
    objId = roller.loadURDF(urdfObj, objPos, objOrn,
                            globalScaling=globalScaling)

    # Add constraint to movable roller (upper)
    cid = pb.createConstraint(
      rollerID2, -1, -1, -
      1, pb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], init_xyz / 100
    )

    # Save variables
    self.roller = roller

    self.rollerID1, self.rollerPos1, self.rollerOrn1 = rollerID1, rollerPos1, rollerOrn1
    self.rollerID2, self.rollerPos2, self.rollerOrn2 = rollerID2, rollerPos2, rollerOrn2
    self.objId, self.objPos, self.objOrn = objId, objPos, objOrn
    self.cid = cid

  def reset(self):
    """
    Reset environment
    """
    pb.resetBasePositionAndOrientation(self.objId, self.objPos, self.objOrn)
    pb.resetBasePositionAndOrientation(
      self.rollerID2, self.rollerPos2, self.rollerOrn2
    )
    # reset xyz position of upper roller
    self.xyz = [0, 0, self.grasp_z]
    pb.changeConstraint(self.cid, self.xyz, maxForce=5)
    for i in range(10):
      pb.stepSimulation()
    self.roller.update()
    self.error = 0
    self.logs = {"touch": [], "vision": [], "pos": [], "goal": None}
    # reset goal
    self.goal = np.random.uniform([-0.7, -0.7], [0.7, 0.7])
  
  def step(self, action):
    pb.changeConstraint(self.cid, self.xyz, maxForce=5)
    self.step_sim()
    pos = self.pose_estimation(self.color[1], self.depth[1])
    self.logs["pos"].append(pos)
    self.xyz[:2] += action
    rew = 0
    for _ in range(self.skipFrame):
      self.xyz[:2] += action
      self.step_sim(render=False)
      r = self.reward_fn(pos, self.goal, action, xyz=self.xyz)
      rew += r
    r = self.reward_fn(pos, self.goal, action, xyz=self.xyz)
    rew += r
    return {'pos':pos, 'goal': self.goal}, rew, False, {}

  def pose_estimation(self, color, depth):
    """
    Estimate location of the ball
    For simplicity, using depth to get the ball center. Can be replaced by more advanced perception system.
    """
    ind = np.unravel_index(np.argmax(depth, axis=None), depth.shape)
    maxDepth = depth[ind]
    if maxDepth < 0.0005:
      return None
    center = np.array(
      [ind[0] / self.tactoResolution[1], ind[1] / self.tactoResolution[0]]
    )
    return center

  def step_sim(self, render=True):
    """
    Step simulation, sync with tacto simulator

    Args:
        render: whether render tactile imprints, Default: True
    """
    # Step in pybullet
    pb.stepSimulation()
    if not (render):
      return
    st = time.time()
    self.roller.update()
    self.color, self.depth = self.roller.render()
    st = time.time()
    if self.recordLogs:
      self.logs["touch"].append([self.color.copy(), self.depth])
      self.visionColor, self.visionDepth = self.cam.get_image()
      self.logs["vision"].append([self.visionColor, self.visionDepth])
    if self.visTacto:
      color1 = self.color[1].copy()
      x0 = int(self.goal[0] * self.tactoResolution[1])
      y0 = int(self.goal[1] * self.tactoResolution[0])
      color1[x0 - 4: x0 + 4, y0 - 4: y0 + 4, :] = [255, 255, 255]
      self.color[1] = color1
      self.roller.updateGUI(self.color, self.depth)

  def reward_fn(self, state, goal, vel, xyz = None):
    if xyz is None:
      xyz = [0, 0, 0]
    return 1 if state is None else np.sum((state - goal) ** 2) ** 0.5

if __name__ == "__main__":
  env = RollingEnv(skipFrame=2)
  env.reset()
  for _ in range(1000):
    env.step(np.zeros(2))