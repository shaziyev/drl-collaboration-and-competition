# -*- coding: utf-8 -*-
"""
Based on Hiroyuki.Konno DDPG agent version
"""
from model import Actor, Critic

import random
import numpy as np
from collections import deque, namedtuple
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

class DdpgAgent():
    def __init__(self, config, seed, device="cpu"):
        self.seed = seed
        
        # -- Set environment
        self.action_size = config["env"]["action_size"]
        self.env = config["env"]["simulator"]
        self.brain_name = config["env"]["brain_name"]
        self.num_agents = config["env"]["num_agents"]


        # -- Construct Actor/Critic models
        self.actor_local = Actor(config["env"]["state_size"], config["env"]["action_size"], seed, config["actor"]["hidden_layers"]).to(device)
        self.actor_target = Actor(config["env"]["state_size"], config["env"]["action_size"], seed, config["actor"]["hidden_layers"]).to(device)
        self.checkpoint = {"state_size":config["env"]["state_size"],
                           "action_size":config["env"]["action_size"],
                           "hidden_layers":config["actor"]["hidden_layers"],
                           "state_dict":self.actor_local.state_dict()}
        
        self.critic_local = Critic(config["env"]["state_size"], config["env"]["action_size"], seed, config["critic"]["hidden_layers"]).to(device)
        self.critic_target = Critic(config["env"]["state_size"], config["env"]["action_size"], seed, config["critic"]["hidden_layers"]).to(device)
        
        # -- Configure optimizer
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config["learning"]["lr_actor"])
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config["learning"]["lr_critic"])

        self.optimizer_lr_decay = config["learning"]["lr_decay"]["activate"]
        self.actor_optimizer_lr_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                                      step_size=config["learning"]["lr_decay"]["actor_step"],
                                                                      gamma=config["learning"]["lr_decay"]["actor_gamma"])
        self.critic_optimizer_lr_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                       step_size=config["learning"]["lr_decay"]["critic_step"],
                                                                       gamma=config["learning"]["lr_decay"]["critic_gamma"])
        
        # -- Set learning parameters
        self.batch_size = config["learning"]["batch_size"]
        self.buffer_size = config["learning"]["buffer_size"]
        self.discount = config["learning"]["discount"]
        self.max_t = config["learning"]["max_t"]
        self.tau = config["learning"]["soft_update_tau"]
        self.learn_every_n_steps = config["learning"]["learn_every_n_steps"]
        self.num_learn_steps = config["learning"]["num_learn_steps"]
        self.checkpointfile = config["learning"]["checkpointfile"]
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed, device)
                
        self.device=device
        
        self.add_noise = True
        self.ou_noise = OUNoise(self.action_size, seed)
        
        self.hard_copy(self.actor_local, self.actor_target)
        self.hard_copy(self.critic_local, self.critic_target)
        self.step_ctr = 0
        
    def steps(self):
        if self.optimizer_lr_decay:
            self.actor_optimizer_lr_scheduler.step()
            self.critic_optimizer_lr_scheduler.step()
            
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.ou_noise.reset()
        state = env_info.vector_observations
        score = np.zeros(self.num_agents)
        self.step_ctr = 0
        while True:
            action = self.act(state)
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done   
            self.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                break
        
        return score, self.step_ctr
            
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_ctr += 1
        if len(self.memory) > self.batch_size and self.step_ctr % self.learn_every_n_steps == 0:
            for _ in range(self.num_learn_steps):
                self.learn()
            
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()   # set train= False
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()  # set back train=True
        
        if self.add_noise:
            action += self.ou_noise.sample()
        
        return np.clip(action, -1, 1)
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample_random()
        
        # -------------------- Update Critic -----------------------------
        # Get predicted next-state actions and Q values from target model
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions).detach()
        
        # Compare Q targets for current states (y_i)
        Q_targets = rewards + (self.discount * Q_targets_next * (1 - dones))
        
        # Compute Critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # -------------------- Update Actor -----------------------------
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------ Update Target Networds --------------------
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param + (1.0-self.tau)*target_param)        
        
    def hard_copy(self, model_a, model_b):
        """ copy model_a to model_b """
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            param_b.data.copy_(param_a)
            
    def reset(self):
        self.actor_local.reset_parameters()
        self.actor_target.reset_parameters()
        self.critic_local.reset_parameters()
        self.critic_target.reset_parameters()
        
    def set_lr(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            self.actor_optimizer
 
    def save_model(self):
        torch.save(self.checkpoint, self.checkpointfile)
        
    def add_noise_on_act(self, nois_on_act):
        """ When nois_on_act is True, OU noise is added in act() """
        self.add_noise = nois_on_act

        
class ReplayBuffer:
    """ Fixed size bufffer to store experience tuples"""
    
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"]
                                     )
        self.seed = random.seed(seed)
        self.device = device
        
    def add(self, states, actions, rewards, next_states, dones):
        assert states.shape[0] == actions.shape[0] and \
            actions.shape[0] == len(rewards) and \
            len(rewards) == next_states.shape[0] and \
            next_states.shape[0] == len(dones)
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        
    def sample_random(self):
        """
        Rondomly sample a batch of experiences from memory and put
        that in device (CPU/GPU)
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        return len(self.memory)

    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.01):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state