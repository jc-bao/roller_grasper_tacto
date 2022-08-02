from typing import Tuple, List
import numpy as np
import open3d as o3d
from copy import deepcopy


class RollerSLAM:
  def __init__(self, width: int, height: int, focal_width: int, focal_height: int, voxel_size: float, matching_num: int) -> None:
    # camera paramters
    self.width = width
    self.height = height
    self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
      width, height, focal_width, focal_height, width//2, height/2)

    # SLAM paramteters
    self.voxel_size = voxel_size
    self.matching_num = matching_num
    self.max_correspondence_distance_fine = voxel_size*1.5
    self.max_correspondence_distance_coarse = voxel_size*15

    # visualization paramteters
    self.o3dvis = self.create_o3dvis(render_size=1024)
    self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
      size=0.02)


  def depth2pcd(self, depth_image: np.ndarray, world2cam_trans: np.ndarray) -> o3d.geometry.PointCloud:
    """return 

    Args:
        depth_image (np.ndarray): _description_
        world2cam_trans (np.ndarray): _description_

    Returns:
        o3d.geometry.PointCloud: point cloud in world frame
    """
    sensor_far = depth_image.max()
    depth_01 = depth_image / sensor_far
    depth_01[depth_01 > 0.98] = 0
    depth_i16 = o3d.geometry.Image((depth_01*65535).astype(np.uint16))
    assert (self.height, self.width) == depth_image.shape, 'depth image shape is not equal to camera resolution'

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
      depth_i16, self.pinhole_camera_intrinsic, world2cam_trans,
      depth_scale=65535/sensor_far, project_valid_depth_only=True)
    pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
    pcd_down.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    assert pcd_down is not None, 'point cloud fail to generate'
    return pcd_down

  def merge_pcds(self, old_pcds: List[o3d.geometry.PointCloud], new_pcd: o3d.geometry.PointCloud, obj2world_trans: List[np.ndarray], new_obj2world_trans: np.ndarray, pose_graph_in_obj: o3d.pipelines.registration.PoseGraph) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray], o3d.pipelines.registration.PoseGraph]:
    """merge old point clouds into the new one and return new point cloud and esitimated new orientation of the new point cloud

    Args:
        old_pcds (o3d.geometry.PointCloud): old point clouds
        new_pcd (o3d.geometry.PointCloud): new point cloud
        old_trans (np.ndarray): old transition of the object
        delta_trans(np.ndarray): relative transition to the object

    Returns:
        Tuple[o3d.geometry.PointCloud, np.ndarray]: new point clouds, new orientation
    """
    old_pcds_num = len(old_pcds)
    source_id = old_pcds_num
    for i in range(min(old_pcds_num, self.matching_num)):
      target_id = old_pcds_num - i - 1
      old2new_trans = obj2world_trans[target_id] @ np.linalg.inv(new_obj2world_trans)
      icp_coarse = o3d.pipelines.registration.registration_icp(
        new_pcd, old_pcds[target_id], self.max_correspondence_distance_coarse, 
        old2new_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
      icp_fine = o3d.pipelines.registration.registration_icp(
        new_pcd, old_pcds[target_id], self.max_correspondence_distance_fine,
        icp_coarse.transformation, # Note: tar2src
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
      information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        new_pcd, old_pcds[target_id], self.max_correspondence_distance_fine,
        icp_fine.transformation)
      if i == 0:
        new_obj2world_trans = np.linalg.inv(icp_fine.transformation) @ obj2world_trans[target_id]
        obj2world_trans.append(new_obj2world_trans)
        pose_graph_in_obj.nodes.append(
          o3d.pipelines.registration.PoseGraphNode(
            new_obj2world_trans))

        pose_graph_in_obj.edges.append(
          o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   icp_fine.transformation,
                                                   information_icp,
                                                   uncertain=False))
      else:
        pose_graph_in_obj.edges.append(
          o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   icp_fine.transformation,
                                                   information_icp,
                                                   uncertain=True))
    old_pcds.append(new_pcd)
    return old_pcds, obj2world_trans, pose_graph_in_obj

  def create_o3dvis(self, render_size: int = 512):
    o3dvis = o3d.visualization.Visualizer()
    o3dvis.create_window(
      width=render_size, height=render_size, visible=False)
    o3dvis.get_render_option().background_color = (0.9, 0.9, 0.9)
    return o3dvis

  def change_o3dvis(self, o3dvis) -> None:
    o3dvis.get_view_control().set_lookat(np.array([0, 0, 0]))
    o3dvis.get_view_control().set_front(np.array([1, 1, 1]))
    o3dvis.get_view_control().set_up(np.array([0, 0, 1]))

  def vis_pcds(self, pcds: List[o3d.geometry.PointCloud], trans: List[np.ndarray]) -> np.ndarray:
    obj2world_trans = trans[-1]
    world2obj_trans = np.linalg.inv(obj2world_trans)
    self.o3dvis.add_geometry(
      deepcopy(self.world_frame).transform(world2obj_trans))
    for pcd, tr in zip(pcds, trans):
      world2obj_trans = np.linalg.inv(tr)
      self.o3dvis.add_geometry(deepcopy(pcd).transform(world2obj_trans))
    self.change_o3dvis(self.o3dvis)
    color = self.o3dvis.capture_screen_float_buffer(do_render=True)
    color = (np.asarray(color)*256).astype(np.uint8)
    self.o3dvis.clear_geometries()
    return color
