import pytest
import dill
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from roller_slam.slam import RollerSLAM


@pytest.fixture
def rollerSLAM():
  return RollerSLAM(
    width=256,
    height=64,
    focal_width=64//2,
    focal_height=256//2,
    voxel_size=0.00002,
    matching_num=3)


@pytest.fixture
def data():
  with open('assets/data_usb_x_rdep2e-3.pkl', 'rb') as f:
    data = dill.load(f)
  '''
      'left_cam'
      'world2leftcam_trans'
      'right_cam': list(),
      'world2rightcam_trans'
      'obj_trans'
      'real_delta_trans'
      'estimated_delta_trans'
  '''
  return data

def test_depth_img(data):
  depth = data['left_cam'][0]
  far = depth.max()
  depth_01 = (depth / far).astype(np.float32)
  depth_01[depth_01 > 0.98] = 0
  depth_8 = (depth_01*256).astype(np.uint8)
  Image.fromarray(depth_8, mode='L').convert('RGB').save('results/test_depth_img.jpg')

def test_depth2pcd(rollerSLAM, data):
  pcds = []
  colors = []
  obj2world_trans = []
  num_data = min(2, len(data['left_cam']))
  for i in range(num_data):
    for cam in ['left', 'right']:
      depth_img = data[f'{cam}_cam'][i]
      world2cam_trans = data[f'world2{cam}cam_trans'][i]
      pcd = rollerSLAM.depth2pcd(depth_img, world2cam_trans)
      pcds.append(pcd)
      obj2world_trans.append(data['obj_trans'][i])
    color = rollerSLAM.vis_pcds(pcds, obj2world_trans)
    colors.append(color)
  color_Img = [Image.fromarray(c) for c in colors]
  color_Img[0].save('results/test_depth2pcd.gif', save_all=True, append_images=color_Img[1:])

def test_merge_pcds(rollerSLAM, data):
  # load pcds from depth image
  pose_graph = o3d.pipelines.registration.PoseGraph()
  num_pcds = min(70, len(data['left_cam']))
  old_pcds = []
  colors = []
  closed_loop = False
  for i in tqdm(range(num_pcds)):
    new_pcd = o3d.geometry.PointCloud()
    for cam in ['left', 'right']:
      depth_img = data[f'{cam}_cam'][i]
      world2cam_trans = data[f'world2{cam}cam_trans'][i]
      new_pcd += rollerSLAM.depth2pcd(depth_img, world2cam_trans)
    if i == 0:
      old_pcds.append(new_pcd)
      pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    else:
      last_old_trans = np.array(pose_graph.nodes[-1].pose)
      new_obj2world_trans = data['estimated_delta_trans'][i-1] @ last_old_trans

      # color_init = rollerSLAM.vis_pcds([*old_pcds, new_pcd], [*obj2world_trans, new_obj2world_trans]) 
      # color_truth = rollerSLAM.vis_pcds([*old_pcds, new_pcd], 
      #   [np.linalg.inv(data['obj_trans'][i])@t for t in data['obj_trans'][:i]])

      # Key function
      old_pcds, pose_graph = rollerSLAM.merge_pcds(old_pcds, new_pcd, new_obj2world_trans, pose_graph)

      color_end = rollerSLAM.vis_pcds(old_pcds, 
        [np.linalg.inv(last_old_trans)@ n.pose for n in pose_graph.nodes])
      # color = np.concatenate([color_truth, color_end], axis=1)
      colors.append(color_end)

      # detect if we are going to merge graph
      obj_rot_angle = np.linalg.norm(R.from_matrix(last_old_trans[:3,:3]).as_rotvec())
      if abs(obj_rot_angle-np.pi) < np.pi/20 and not closed_loop:
        print('adding constrains...')
        rollerSLAM.add_graph_edge(pose_graph, old_pcds, i, list(range(10)))
        print('optimize...')
        pose_graph = rollerSLAM.optimize_graph(pose_graph)
        closed_loop = True
      if closed_loop:
        print('adding constrains...')
        start = i - 49 # This is hardcoded, Remove it later
        rollerSLAM.add_graph_edge(pose_graph, old_pcds, i, list(range(start,start+10)))
        print('optimize...')
        pose_graph = rollerSLAM.optimize_graph(pose_graph)

  pcds = rollerSLAM.create_pcd(old_pcds, [np.linalg.inv(last_old_trans)@ n.pose for n in pose_graph.nodes])
  o3d.io.write_point_cloud('results/final_pcd.pcd', pcds)
  colors_img = [Image.fromarray(c) for c in colors]
  # visualize result
  colors_img[0].save('results/test_merge_pcds.gif', save_all=True, append_images=colors_img[1:])