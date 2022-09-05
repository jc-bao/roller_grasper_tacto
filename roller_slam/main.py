from roller_slam.bo import run_bo
import ray
import numpy as np
import pandas as pd
import tqdm
from functools import partialmethod

@ray.remote
def main(x):
  init_explore_section = x[:3]
  UCB_alpha = x[3]
  results = run_bo(init_explore_section, UCB_alpha)
  print('====finished====')
  return results
  

if __name__ == '__main__':
  ray.init(num_cpus=128)

  angle_step = np.pi/6
  section_step = 0.025

  UCB_alpha = np.array([-1, 1, 500, 50000])

  theta = np.arange(-np.pi, np.pi, angle_step)
  phi = np.arange(-np.pi/2, np.pi/2, angle_step)
  section_disp = np.arange(0, 0.001, section_step)

  xx = np.stack(np.meshgrid(theta, phi, section_disp, UCB_alpha), axis=-1)
  xx_flatten = xx.reshape(-1, xx.shape[-1])

  results = [main.remote(x) for x in xx_flatten]
  results = np.array(ray.get(results))
  err = results[:, 0]
  step = results[:, 1]
  data = {
    'theta': xx_flatten[..., 0], 
    'phi': xx_flatten[..., 1], 
    'disp': xx_flatten[..., 2], 
    'alpha': xx_flatten[..., 3],
    'err': err, 
    'step': step, 
    'success': (err > -0.035).astype(np.float32), 
  }
  data = pd.DataFrame(data)
  data.to_csv('../test/results/data.csv')
  print(err, step)