import numpy as np
def inverse(position,length):
  pose = np.array([0,0,0,0,0,0])
  if (np.linalg.norm(position)>3*length):
    print("Position out of range!")
    return pose 
  
  return pose