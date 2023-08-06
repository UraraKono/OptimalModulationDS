import sys
import matplotlib.pyplot as plt

sys.path.append('../ds_mppi/')
import numpy as np
import torch
import time
t0 = time.time()
params = {'device': 'cpu', 'dtype': torch.float32}
p_min = -1.1*np.array([-10, -10])
p_max = -1.1*np.array([10, 10])

# number of samples
N_JPOS = 4000
N_PPOS = 500
# generate random joint positions
rand_jpos = np.random.uniform(p_min, p_max, (N_JPOS, 2)) #(4000,2)
rand_jpos = torch.tensor(rand_jpos).to(**params)
# print(rand_jpos.shape)
# compute forward kinematics for pregenerated joint positions
t0 = time.time()
# start main loop
data = torch.zeros(N_JPOS * (N_PPOS+50), 5)  # joint position, point position, distances per link
k = 0
for i in range(N_JPOS):
    print(i)
    rand_ppos = torch.tensor(np.random.uniform(p_min, p_max, (N_PPOS, 2))) #(500,2)
    rnd_near = np.random.uniform(0.1*p_min, 0.1*p_max, (50, 2)) #(50,2)
    rand_ppos2 = rand_jpos[i] + torch.tensor(rnd_near).to(**params) #(1,2) + (50,2) = (50,2)
    rand_ppos = torch.vstack((rand_ppos, rand_ppos2)) #(550,2)
    dist = torch.norm(rand_ppos - rand_jpos[i], 2, 1) #(550)
    data[k:k+rand_ppos.shape[0]] = torch.hstack((rand_jpos[i].repeat(rand_ppos.shape[0], 1), rand_ppos, dist.reshape(-1, 1))) #(550,5)
    k = k+rand_ppos.shape[0]

    # print(rand_ppos.shape)
    # print(rnd_near.shape)
    # print(rand_ppos2.shape)
    # print(rand_ppos.shape)
    # print(dist.shape)
    # print(data[k:k+rand_ppos.shape[0]].shape)
    # print(rand_jpos[i].repeat(rand_ppos.shape[0], 1).shape)
print("time: %4.3f s" % (time.time() - t0))

torch.save(data, 'datasets/2d_toy_data.pt')
