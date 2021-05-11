import gym
import ray
from ray import tune
from ray.rllib import agents
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set()

ray.init()
config = {'gamma': 0.999,
          'lr': 0.01,
          "n_step": 1000,
          'num_workers': 3,
          'monitor': False}
trainer2 = agents.dqn.DQNTrainer(env='LunarLander-v2', config=config)
results2 = trainer2.train()

# TODO; Save train model

env = gym.make("LunarLander-v2")
# Instantiate enviroment with default parameters
observation = env.reset()
for step in range(300):
    env.render() # Show agent actions on screen
    observation, reward, done, info = env.step(trainer2.compute_action(observation)) # Gets t action
    if done:
        time.sleep(3)
        break
env.close()