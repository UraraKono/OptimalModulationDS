import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import numpy as np
sys.dont_write_bytecode = False
import copy

sys.path.append('../functions/')
from MPPI_toy import *
sys.path.append('../../mlp_learn/')
from sdf.robot_sdf import RobotSdfCollisionNet

params = {'device': 'cpu', 'dtype': torch.float32}

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
DOF = 2
L = 3

# Load nn model
s = 256
n_layers = 5
skips = []
fname = '%ddof_sdf_%dx%d_mesh.pt' % (DOF, s, n_layers)
fname = '%ddof_sdf_%dx%d_toy.pt' % (DOF, s, n_layers)
if skips == []:
    n_layers -= 1
nn_model = RobotSdfCollisionNet(in_channels=DOF+2, out_channels=1, layers=[s] * n_layers, skips=skips)
nn_model.load_weights('../../mlp_learn/models/' + fname, params)
nn_model.model.to(**params)
nn_model.model_jit = nn_model.model
nn_model.model_jit = torch.jit.script(nn_model.model_jit)
nn_model.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
nn_model.update_aot_lambda()

nn_model2 = RobotSdfCollisionNet(in_channels=DOF+2, out_channels=1, layers=[s] * n_layers, skips=skips)
nn_model2.load_weights('../../mlp_learn/models/' + fname, params)
nn_model2.model.to(**params)
nn_model2.model_jit = nn_model.model
nn_model2.model_jit = torch.jit.script(nn_model.model_jit)
nn_model2.model_jit = torch.jit.optimize_for_inference(nn_model.model_jit)
nn_model2.update_aot_lambda()

# q_min = -10 * torch.ones(DOF).to(**params)
# q_max = 10 * torch.ones(DOF).to(**params)
#
q_min = torch.tensor([-4, -0.5]).to(**params)
q_max = torch.tensor([9, 4.5]).to(**params)

# q_min = torch.tensor([0, -1.25]).to(**params)
# q_max = torch.tensor([4, 1.25]).to(**params)

# generate meshgrid within q_min, q_max
n_mg = 50
points_grid = torch.meshgrid([torch.linspace(q_min[i], q_max[i], n_mg) for i in range(DOF)])
q_tens = torch.stack(points_grid, dim=-1).reshape(-1, DOF)

# obstacles
# generate arc consisting of spheres of radius 1
#concave case
t = torch.linspace(-np.pi/3, np.pi/3, 20)
c = torch.tensor([0, 0])
#convex case
# t1 = torch.linspace(2*np.pi/3, np.pi, 10)
# t2 = torch.linspace(-np.pi, -2*np.pi/3, 10)
# t = torch.hstack([t1, t2])
# c = torch.tensor([3, 0])

r_arc = 3
r_sph = 1
obs_arr = torch.vstack([c[0]+r_arc * torch.cos(t),
                        c[1]+r_arc * torch.sin(t),
                        r_sph*torch.ones(t.shape[0])]).transpose(0, 1)
obs_tens = torch.tensor(obs_arr).to(**params)

# obs_tens = torch.tensor([[0, 2, 0, 1], [2, 1, 0, 2], [2, -1, 0, 2], [0, -2, 0, 1]]).to(**params)
# obs_tens = obs_tens[0:1]
# input tensor for nn (using repeat_interleave)
nn_input = torch.hstack((q_tens.tile(obs_tens.shape[0], 1), obs_tens.repeat_interleave(q_tens.shape[0], 0)))
distances = nn_model.model(nn_input[:, 0:DOF+2]).detach()
distances = distances.min(dim=1).values # find closest link in each case
distances -= nn_input[:, -1] # substract radii
distances = distances.reshape(obs_tens.shape[0], q_tens.shape[0]).transpose(-1,0) # reshape obstacle-like
distances = distances.min(dim=1).values # find closest obstacle in each case
distances = distances.reshape(n_mg, n_mg) # reshape grid-like

# plot contour of distances
plt.figure()

# DS
A = torch.tensor([[-1, 0], [0, -1]]).to(**params)
attractor = torch.tensor([8, 0]).to(**params)
# calculate ds flow
ds_flow = A @ (q_tens - attractor).transpose(0, 1)
# plot streamplot
# plt.streamplot(points_grid[0][:, 0].numpy(),
#                points_grid[1][0, :].numpy(),
#                ds_flow[0].reshape(n_mg, n_mg).numpy().T,
#                ds_flow[1].reshape(n_mg, n_mg).numpy().T,
#                density=2, color='b', linewidth=0.5, arrowstyle='->')
#
# plt.plot(attractor[0], attractor[1], 'r*', markersize=10)
#plt.show()

q_0 = torch.tensor([-5, 0]).to(**params)
mppi_step = MPPI(q_0, attractor, torch.zeros(4, 4), obs_tens, 0.1, 1, q_tens.shape[0], A, 0, nn_model, 5)
ds_flow = torch.zeros(q_tens.shape).to(**params)
mppi_step.q_cur = q_tens
_, _, _, _ = mppi_step.propagate()
ds_flow = mppi_step.qdot#[0, :]

# for i, point in enumerate(q_tens):
#     mppi_step.q_cur = point
#     _, _, _, _ = mppi_step.propagate()
#     ds_flow[i] = mppi_step.qdot[0, :]
#     print(i)

plt.streamplot(points_grid[0][:, 0].numpy(),
               points_grid[1][0, :].numpy(),
               ds_flow[:,0].reshape(n_mg, n_mg).numpy().T,
               ds_flow[:,1].reshape(n_mg, n_mg).numpy().T,
               density=1, color='b', linewidth=0.5, arrowstyle='->', broken_streamlines= True)

trajs_init = torch.tensor(np.linspace([-10,-10], [-10, 10], 9)).to(**params)
trajs = []

mppi_traj = MPPI(trajs_init[0], attractor, torch.zeros(4, 4), obs_tens, 0.01, 3000, trajs_init.shape[0], A, 0, nn_model2, 5)
policy = torch.load('toy_policy.pt')
mppi_traj.Policy.update_with_data(policy)
mppi_traj.Policy.alpha_s *= 0
mppi_traj.Policy.sample_policy()  # samples a new policy using planned means and sigmas

#for point in trajs_init:
#mppi_traj.q_cur = torch.tensor(point).to(**params)
mppi_traj.q_cur = trajs_init
trajs, _, _, _ = mppi_traj.propagate()
    #trajs.append(traj)

for traj in trajs:
    plt.plot(traj[:, 0], traj[:, 1], color=[0.85, 0.32, 0.1], linewidth=1.5)
plt.plot(attractor[0], attractor[1], 'r*', markersize=10)

#finally place obstacles
# custom listed colormap
carr_obs = np.linspace([.1, .1, 1, 1], [.5, .5, 1, 1], 256)
carr_obs[-1] = [1, 1, 1, 0]

plt.contourf(points_grid[0], points_grid[1], distances, levels=1000, cmap=ListedColormap(carr_obs), vmax=0.)
# contour zero lvl
plt.contour(points_grid[0], points_grid[1], distances, levels=[0], colors='k')
plt.gca().set_xlim([q_min[0], q_max[0]])
plt.gca().set_ylim([q_min[1], q_max[1]])

plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('ds_mod_4a.png', dpi = 600)
plt.show()
