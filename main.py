# Module: main.py
# Created on: 4/25/17
# Author: Jake Sacks

import config
import ddp
import dynamics
import network
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# global and MPC variables
config.init() # initialize global variables
x_0 = np.matrix([0,0,np.pi,0]).T # initial state
x_f = np.matrix([0,0,0,0]).T # final state
mpc_N = 200
mpc_mode = False
online_training = False

# vectors to keep track of results
x_vect = np.matrix(np.zeros([config.num_states, mpc_N+1]))
x_vect[:, 0] = x_0
x_pred = x_vect.copy()
u_vect = np.matrix(np.zeros([config.num_inputs, mpc_N]))
dx_vect = np.matrix(np.zeros([config.num_states, mpc_N]))
cost_vect = np.zeros([mpc_N])

# load the neural network
net = network.load(None)

# MPC main loop
for i in range(0, mpc_N):

    print('Iteration: {0}'.format(i))
    (x_bar, u_bar, cost) = ddp.run(x_0, x_f)

    if (mpc_mode):
        # interact with system
        u = u_bar[:,0]
        x = x_bar[:,0]

        dx = dynamics.nonlinear(x, u)
        x_0 += dx * config.dt

        # train network
        if (online_training):
            x_tilde = np.concatenate((x, u / 100))
            net = network.train(net, x_tilde.T, dx.T)

        # save terms in vector
        x_vect[:,i+1] = x_0
        u_vect[:,i] = u
        dx_vect[:,i] = dx
        cost_vect[i] = cost[0]
    else:
        x_pred = np.matrix(np.zeros([config.num_states, config.tf]))
        x_pred[:,0] = x_0
        for j in range(0, config.tf-1):
            x_tilde = np.concatenate((x_pred[:, j], u_bar[:, j]/100))
            dx = net.predict(x_tilde.T).T
            x_pred[:,j+1] = x_pred[:, j] + dx * config.dt
        break

# evaluate network on entire MPC trajectory
if (mpc_mode):
    for i in range(0, mpc_N):
        u = u_vect[:,i]
        x = x_pred[:,i]

        x_tilde = np.concatenate((x, u / 100))
        dx_nn = net.predict(x_tilde.T).T
        x_pred[:, i + 1] = x_pred[:, i] + dx_nn * config.dt
        dx_vect[:, i] = dynamics.nonlinear(x, u)

    x_tilde = np.concatenate((x_pred[:, 0:mpc_N], u_vect/100))
    network.evaluate(net, x_tilde.T, dx_vect.T)

# save the network
if (online_training):
    filename = 'weights/weights_iter1000_100traj_100epoch_mpc'+str(mpc_N)+'.json'
    network.save(net, filename)

# plot the results
# if running normal DDP
if (mpc_mode == False):
    t = np.arange(0, config.tf) * config.dt

    plt.figure(1)
    plt.subplot(2,2,1), plt.plot(t, x_bar[0,:].T, t, x_pred[0,:].T), plt.xlabel('time (s)'), plt.ylabel('x (m)'), plt.title('x vs time')
    plt.subplot(2,2,2), plt.plot(t, x_bar[1,:].T, t, x_pred[1,:].T), plt.xlabel('time (s)'), plt.ylabel('dx/dt (m/s)'), plt.title('dx/dt vs time')
    plt.subplot(2,2,3), plt.plot(t, x_bar[2,:].T, t, x_pred[2,:].T), plt.xlabel('time (s)'), plt.ylabel(r'$\theta$ (rad)'), plt.title(r'$\theta$ vs time')
    plt.subplot(2,2,4), plt.plot(t, x_bar[3,:].T, t, x_pred[3,:].T), plt.xlabel('time (s)'), plt.ylabel(r'd$\theta$/dt (rad/s)'), plt.title(r'd$\theta$/dt vs time')

    plt.figure(2)
    plt.subplot(2,1,1), plt.plot(t[0:config.tf-1], u_bar[0,:].T), plt.xlabel('time (s)'), plt.ylabel('u (N)'), plt.title('u vs time')
    plt.subplot(2,1,2), plt.plot(cost), plt.xlabel('iteration'), plt.ylabel('cost'), plt.title('cost per iteration')

# if running in MPC mode
else:
    t = np.arange(0, mpc_N+1) * config.dt
    plt.figure(1)
    plt.subplot(2,2,1), plt.plot(t, x_vect[0,:].T, t, x_pred[0,:].T, 'ro'), plt.xlabel('time (s)'), plt.ylabel('x (m)'), plt.title('x vs time')
    plt.subplot(2,2,2), plt.plot(t, x_vect[1,:].T, t, x_pred[1,:].T, 'ro'), plt.xlabel('time (s)'), plt.ylabel('dx/dt (m/s)'), plt.title('dx/dt vs time')
    plt.subplot(2,2,3), plt.plot(t, x_vect[2,:].T, t, x_pred[2,:].T, 'ro'), plt.xlabel('time (s)'), plt.ylabel(r'$\theta$ (rad)'), plt.title(r'$\theta$ vs time')
    plt.subplot(2,2,4), plt.plot(t, x_vect[3,:].T, t, x_pred[3,:].T, 'ro'), plt.xlabel('time (s)'), plt.ylabel(r'd$\theta$/dt (rad/s)'), plt.title(r'd$\theta$/dt vs time')

    plt.figure(2)
    plt.subplot(2,1,1), plt.plot(t[0:mpc_N], u_vect[0,:].T), plt.xlabel('time (s)'), plt.ylabel('u (N)'), plt.title('u vs time')
    plt.subplot(2,1,2), plt.plot(cost_vect), plt.xlabel('iteration'), plt.ylabel('cost'), plt.title('cost per iteration')

# display the plot
plt.show()
