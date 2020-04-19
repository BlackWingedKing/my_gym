# imports
import sys
import gym
from gym import spaces
from gym import envs

# registry
# print(envs.registry.all())
# exit()

# # spaces test
# space = spaces.Discrete(8)
# x = space.sample()
# print(x)

# create an envt 
env = gym.make('Acrobot-v1')
# some envt attributes
print('printing action and obs spaces of type: Space')
print('action_space:', env.action_space)
print('observation_space:', env.observation_space)
print('observation_space_low:', env.observation_space.low)
print('observation_space_high:', env.observation_space.high)

# run the algo for 20 episodes
for i_episode in range(20):
# intialise it to default states
    observation = env.reset()
    for t in range(100):
        env.render() # returns a true or false value depending upon the display
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
