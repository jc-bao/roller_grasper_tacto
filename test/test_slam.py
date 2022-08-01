import pytest
from roller_slam import slam

def test_start_test():
  assert slam.start_test() == 0