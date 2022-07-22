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

- [ ] Replace Sensor with Roller

### L2

- [ ] Gym Wrapper