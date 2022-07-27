# Roller Grasper v4 Tacto Environment

Based on PyBullet and Tacto. 

## Environment

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
- [ ] Handcrafted policy for reorientation
  - [x] Single rotation handcrafted policy (2022.7.22)
  - [x] attach sensor to fixed joint (2022.7.23)
  - [ ] Compositonal handcrafted policy
- [ ] Reconstruct the object from depth image
  - [ ] With wide camera range
    - [ ] With true object position
    - [ ] With position esitimate from roller angle
  - [ ] With small camera range
  - [ ] With moving camera
- [ ] Reconstruct the object from tactile sensor

### L2

- [x] Gym Wrapper (2022.07.21)
- [x] Roller Control Suite
  - [x] Add roller model to simulation (2022.7.22)
  - [x] Add sensor to simulation (2022.7.22)
- [x] off screen rendering (2022.7.27)

### L3

- [ ] Add blender to simulation

## Details

* roller size: R=0.02m H=0.05m
* this package use scipy, in which quaternion is [x,y,z,w], in pybullet, quaternion is also [x,y,z,w]. but in mujoco, quaternion is [w,x,y,z]
