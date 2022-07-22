# Roller Grasper v4 Tacto Environment

Based on PyBullet and Tacto. 

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

### L2

- [x] Gym Wrapper (2022.07.21)