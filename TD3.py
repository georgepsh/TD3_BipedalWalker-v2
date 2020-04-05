import gym
import pybullet_envs
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from time import perf_counter
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform(item, convert=True):
    res = torch.tensor(item).float()
    return res.to(device)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        s = torch.FloatTensor(batch.state)
        ns = torch.FloatTensor(batch.next_state)
        a = torch.FloatTensor(batch.action)
        r = torch.FloatTensor(batch.reward).unsqueeze(1)
        d = torch.tensor(batch.done).unsqueeze(1)
        return s, ns, a, r, d

    def __len__(self):
        return len(self.memory)


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


class Critic(nn.Module):
    def __init__(self, l1_size, l2_size, state_dim, act_dim):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + act_dim, l1_size)
        self.linear2 = nn.Linear(l1_size, l2_size)
        self.linear3 = nn.Linear(l2_size, 1)

        self.linear4 = nn.Linear(state_dim + act_dim, l1_size)
        self.linear5 = nn.Linear(l1_size, l2_size)
        self.linear6 = nn.Linear(l2_size, 1)

    def forward(self, state, action, only_first=False):
        inp = torch.cat([state, action], dim=1)

        Q1 = F.relu(self.linear1(inp))
        Q1 = F.relu(self.linear2(Q1))
        Q1 = self.linear3(Q1)

        if only_first:
            return Q1

        Q2 = F.relu(self.linear4(inp))
        Q2 = F.relu(self.linear5(Q2))
        Q2 = self.linear6(Q2)

        return Q1, Q2


class TD3:
    def __init__(self, state_dim, act_dim, max_action, hid_size1=256, hid_size2=256, \
                    gamma=0.97, polyak=0.95, noise=0.2, clip=0.5, upd_rate=2, \
                    optim_method=optim.Adam, critic_loss=F.mse_loss):
        self.gamma = gamma
        self.polyak = polyak
        self.noise = noise
        self.clip = clip
        self.upd_rate = upd_rate
        self.max_action = max_action
        self.critic_loss = critic_loss

        self.critic = Critic(hid_size1, hid_size2, state_dim, act_dim)
        self.critic_target = Critic(hid_size1, hid_size2, state_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(hid_size1, hid_size2, state_dim, act_dim, max_action)
        self.actor_target = Actor(hid_size1, hid_size2, state_dim, act_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_optim = optim_method(self.critic.parameters(), lr=3e-4)
        self.actor_optim = optim_method(self.actor.parameters(), lr=3e-4)

        self.iteration = 0
    
    def act(self, state):
        action = self.actor(transform(state))
        return action.data.numpy()
    
    def update(self, memory: ReplayMemory, batch_size):
        if len(memory) < batch_size:
            return

        self.iteration += 1
        states, next_states, actions, rewards, dones = memory.sample(batch_size)
        
        Q1, Q2 = self.critic(states, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.noise).clamp(-self.clip, self.clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            Q_next = torch.min(*self.critic_target(next_states, next_actions))
            Q_target = rewards + (1 - dones) * self.gamma * Q_next
        
        self.update_critic(Q1, Q2, Q_target)
        if self.iteration % self.upd_rate == 0:
            self.update_actor(states)
            self.update_target_networks()
    
    def update_critic(self, Q1, Q2, Q_target):
        loss = self.critic_loss(Q1, Q_target) + self.critic_loss(Q2, Q_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def update_actor(self, state):
        loss = self.actor_loss(state)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def actor_loss(self, state):
        return -self.critic(state, self.actor(state), only_first=True).mean()
        
    def update_target_networks(self):
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, estimate_model, target_model):
        for estimate_param, target_param in zip(estimate_model.parameters(), target_model.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + estimate_param * (1 - self.polyak))

    def save(self, name='walker.pkl'):
        torch.save(self.actor.state_dict(), name)
        print('------Model Saved------')


def test_agent(model: TD3, env_name, episodes=10, render=False):
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
                ep_score += reward
            scores.append(ep_score)

    return sum(scores) / len(scores)

env_name = "BipedalWalker-v2"
env = gym.make(env_name)
n_actions = env.action_space.shape[0]
n_states = env.observation_space.shape[0]
max_action = env.action_space.high[0]
noise = 0.1
max_episodes = 1000
exploration_episodes = 25
memory = ReplayMemory(int(5e4))
batch_size = 512
test_rate = 10
goal = 3000
model = TD3(n_states, n_actions, max_action=max_action, hid_size1=400, hid_size2=300, \
            gamma=0.98, polyak=0.95, noise=0.2, clip=0.5, upd_rate=2, critic_loss=F.smooth_l1_loss)

scores = []
current_best = -float('inf')
for ep in range(1, max_episodes):
    state = env.reset()
    done = False
    ep_score = 0
    start = perf_counter()
    while not done:
        if ep < exploration_episodes:
            action = env.action_space.sample()
        else:
            action = model.act(np.array(state)) + np.random.normal(0, max_action * noise, size=n_actions)
            action = action.clip(-max_action, max_action)
        
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, next_state, reward, int(done))
    
        state = next_state
        ep_score += reward

        if ep >= exploration_episodes:
            model.update(memory, batch_size)

    print(f'{int(ep_score)} at episode {ep}, time: {perf_counter() - start}')
    if ep % test_rate == 0:
        print()
        print('----Testing agent----')
        test_score = test_agent(model, env_name, episodes=10, render=False)
        print('result:', test_score)
        print('----Testing completed----')
        print()
        if test_score > current_best:
            current_best = test_score
            model.save()
    # if current_best > goal:
    #     break
