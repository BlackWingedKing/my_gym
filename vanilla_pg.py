# imports
import sys
import gym
from gym import spaces
from gym import envs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.bernoulli import Bernoulli
import matplotlib
from matplotlib import pyplot as plt

# gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# create an envt 
env = gym.make('CartPole-v1')

# Let's try out a logistic function for the policy
# this is a logistic regression which outputs the policy
# this returns the probability of 1
class policy_net(nn.Module):
    def __init__(self):
        super(policy_net, self).__init__()
        self.fcn1 = nn.Linear(4, 16)
        self.fcn2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fcn1(x))
        x = self.sigmoid(self.fcn2(x))
        return x

# cartpole training
# observation = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# run the algo for n episodes or so update params at the end of the episode
n = 10000
gamma = 1.0
model = policy_net().to(device)
model.train()
optimiser = optim.Adam(model.parameters(), lr=1e-3)

reward_list = []
for i in range(n):
    # episode i like an epoch
    # intialise it to default states
    loss = 0.0
    R = 0.0
    obs = torch.Tensor(env.reset()).to(device).view(-1,4)
    print('episode: ', i, )
    model.zero_grad()
    optimiser.zero_grad()
    for t in range(100):
        # env.render() # returns a true or false value depending upon the display
        p = model.forward(obs) # 0 or 1 force left or right application
        m = Bernoulli(p)
        a = m.sample()
        action = a.bool().item()
        obs, reward, done, info = env.step(action)
        obs = torch.Tensor(obs).to(device).view(-1,4)
        loss+= m.log_prob(a)
        R+= reward*(gamma**t)
        if done:
            # done is a boolean which indicates if it reaches the terminal state
            loss = -1*loss*R
            loss.backward()
            optimiser.step()
            print("finished after {} timesteps".format(t+1))
            reward_list.append(t+1)
            break
env.close()

plt.figure()
plt.plot(reward_list, label='reward')
plt.legend()
plt.show()