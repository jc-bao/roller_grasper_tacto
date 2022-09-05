from roller_slam.bo import run_bo
import ray
import numpy as np
import pandas as pd

@ray.remote
def main(x):
  init_explore_section = x[:3]
  explore_policy = 'bo'
  UCB_alpha = x[3]
  return run_bo(init_explore_section, explore_policy, UCB_alpha)
  

if __name__ == '__main__':
  UCB_alpha = np.array([500])
  theta = np.array([0])
  phi = np.arange(-np.pi/2, np.pi/2, np.pi/4)
  section_disp = np.array([0])
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
    'err': err, 
    'step': step
  }
  data = pd.DataFrame(data)
  data.save_csv('../test/results/data.csv')
  print(err, step)