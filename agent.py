import random
import numpy as np
import os
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("BipedalWalker-v2")
max_action = env.action_space.high[0]
n_actions = env.action_space.shape[0]
n_states = env.observation_space.shape[0]


def transform(item):
    res = torch.tensor(item).float()
    return res.to(device)


class Actor(nn.Module):
    def __init__(self, l1_size, l2_size, state_dim, act_dim, max_action):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, l1_size)
        self.linear2 = nn.Linear(l1_size, l2_size)
        self.linear3 = nn.Linear(l2_size, act_dim)

        self.max_action = max_action
    
    def forward(self, state):
        action = F.relu(self.linear1(state))
        action = F.relu(self.linear2(action))
        return self.max_action * torch.tanh(self.linear3(action))


class Agent:
    def __init__(self):
        self.model = Actor(400, 300, n_states, n_actions, max_action)
        params = torch.load(__file__[:-8] + "/walker.pkl", map_location=device)
        self.model.load_state_dict(params)
        self.model.to(device)
        
    def act(self, state):
        state = transform(np.array(state))
        action = self.model(state)
        return action.cpu().data.numpy()

    def reset(self):
        pass

def test_agent(model: Agent, env_name, episodes=100, render=False):
    env_ = gym.make(env_name)
    scores = []
    with torch.no_grad():
        for _ in range(episodes):
            state = env_.reset()
            if render:
                env_.render()
            ep_score = 0
            done = False
            while not done:
                action = model.act(np.array(state))
                state, reward, done, _ = env_.step(action)
                if render:
                    env_.render()
                ep_score += reward
            scores.append(ep_score)

    return sum(scores) / len(scores)

print(test_agent(Agent(), "BipedalWalker-v2", render=True))
