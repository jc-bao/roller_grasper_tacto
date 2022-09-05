from roller_slam.bo import run_bo
import ray
import numpy as np

@ray.remote
def main(init_explore_section):
  return run_bo(init_explore_section)
  

if __name__ == '__main__':
  phi = np.arange(-np.pi/2, np.pi/2, np.pi/4)
  all_explore_values = np.zeros((phi.shape[0], 3))
  all_explore_values[:,1] = phi
  results = [main.remote(x) for x in all_explore_values]
  results = np.array(ray.get(results))
  err = results[:, 0]
  step = results[:, 1]
  print(err, step)
