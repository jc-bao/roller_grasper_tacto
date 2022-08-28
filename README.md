# Roller Grasper v4 Tacto Environment

Based on PyBullet and Tacto. 

## Environment

### Reconstruction Environment

#### System overview

[new depth image, old point clouds, old orientation, esitimated delta orientation] -`process pcds`-> [new point clouds, new orientation]

Details:

[image data] -`pcd_from_depth(depth_image)`-> [point cloud (in camera frame)] -`pcd_cam2world(pcd, camera_pos)`-> [point cloud (in world frame)] -`merge_pcds(old_pcds, new_pcd, old_orientation, delta_orientation)`-> [point clouds, new_orintation]

#### Demos

* Single sensor case

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4m3v8vii9g20sg0e8b2a.gif)

* Multi-sensor with ICP refinement

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4m8poz16hg20sg0sg1kz.gif)

* Pose estimation with ICP+GO under noisy position esitimatioon

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4may1gbhig20sg0sgkjl.gif)

* Reconstuct object from narrow observation angle

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4mimanv6mg20sg0sgtlz.gif)

### Physical Environment

|Roller Toy Environment| Roller Env (Random explore)|Render gelsight environment|
|-|-|-|
|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4g69amxucj20jo0gs3z1.jpg)|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4g94hyxngg20cw0ac4qq.gif)|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4gsdv7h0hg20ee08uavr.gif)|

|Rolling Action| Pitching Action|Wrist Action|
|-|-|-|
| `act['roll_l_vel'][0] = 1; act['roll_r_vel'][0] = 1` | `act['pitch_l_vel'][0] = 1; act['pitch_r_vel'][0] = 1` | `act['wrist_vel'] = 1` |
|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4g9ti9lolg20cu06w10u.gif)|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4g9v39e8og20cu06wdnm.gif)|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4g9xq9sn3g20cu06wwnf.gif)|

## Handcrafted Policy

|Rolling Action| Pitching Action|Wrist Action|Compositional Action|
|-|-|-|-|
|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4gh2dz1gmg20cu06ktfj.gif)|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4gh2jye7eg20cu06kn47.gif)|![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4gh2psptpg20cu06kgul.gif)|![success-2](https://user-images.githubusercontent.com/60093981/180583278-77c65ff9-ca5c-4ef9-bf88-71f931f4488e.gif)|


## Usage

```
# install a FORTRAN compiler for opto
sudo apt-get install gfortran

# install opto
git clone https://github.com/robertocalandra/opto.git
cd opto
pip install -r requirements.txt
python setup.py install
pip install scipyplot deepdish tacto
```

## TODO List

### L1

- [x] Replace Sensor with Roller (2022.07.21)
- [x] Handcrafted policy for reorientation
  - [x] Single rotation handcrafted policy (2022.7.22)
  - [x] attach sensor to fixed joint (2022.7.23)
  - [x] Compositonal handcrafted policy
- [x] Reconstruct the object from depth image
  - [x] With wide camera range (2022.7.27)
    - [x] With true object position
    - [x] With position esitimate from roller angle
  - [x] With small camera range (2022.7.27)
  - [x] With moving camera (2022.7.27)
- [x] Reconstruct of different shapes (e.g. EDGA dataset) (2022.7.28)
- [x] Wrapper up reconstruction function
  - [x] wrap up functions (2022.7.31)
  - [x] test functions and data wrap up (2022.8.1)
  - [x] solve the problem of phi close to each other. (caused by not clean the debug code timely) (2022.8.27)
  - [ ] solve the problem of be optimistic of unexplored area. (2022.8.27)
    (reason: the section far away from the center, has large phi, thus its variance is constrained. solution: make the initial guess more pessimistic)
- [ ] Efficient way to detect close loop
- [ ] Using ICP to matching points

### L2

- [x] Gym Wrapper (2022.07.21)
- [x] Roller Control Suite
  - [x] Add roller model to simulation (2022.7.22)
  - [x] Add sensor to simulation (2022.7.22)
- [x] off screen rendering (2022.7.27)

### L3

- [ ] Add blender to simulation
- [ ] Reconstruct the object from tactile sensor

## Hardware Deployment

1. get getsight depth image
   1. Requirement: from 0 -> max_depth
2. generate esistimated angle
3. run the test code in test set


Possible Gaps
* The Gel is rounded but our sensor get a plane
* The Gel has less depth


|Constrain with depth and size | Constrain with Gel's shape|
| - | - |
| ![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4u0rhlu7qg20sg0sg4qp.gif) | ![](https://tva1.sinaimg.cn/large/e6c9d24egy1h4u31iks7kg20sg0sg1l1.gif) |

## Details

* roller size: R=0.02m H=0.05m
* this package use scipy, in which quaternion is [x,y,z,w], in pybullet, quaternion is also [x,y,z,w]. but in mujoco, quaternion is [w,x,y,z]
