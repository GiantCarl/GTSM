import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch

tensor_uniform = torch.rand(10)
print("tensor_uniform:", tensor_uniform)

symlog_tensor = torch.sign(tensor_uniform) * torch.log(abs(tensor_uniform) + 1)
print("symlog_tensor:", symlog_tensor)

symexp_tensor = torch.sign(symlog_tensor) *(torch.exp(abs(symlog_tensor))-1)
print("symexp_tensor:", symexp_tensor)

diff = tensor_uniform - symexp_tensor
print("diff:",diff)
