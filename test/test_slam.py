import pytest
import dill
from roller_slam.slam import RollerSLAM

@pytest.fixture
def init_RollerSLAM():
  return RollerSLAM(
    width=256,
    height=64,
    focal_width=256,
    focal_height=256,
    voxel_size=0.00002,
    matching_num=5)
  
@pytest.fixture
def init_data():
  with open('assets/data.pkl', 'rb') as f:
    data = dill.load(f)
  '''
      'left_cam': list(), 
      'right_cam': list(),
      'obj_trans': list(), 
      'real_delta_trans': list(),
      'estimated_delta_trans': list()
  '''
  return data
