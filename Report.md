
# Udacity DRL Collaboration And Competition Project Report

### Algorithm Overview
The agent model is based on the Deep Deterministic Policy Gradients (DDPG) algorithm, which leverages the Actor-Critic networks. Both agents share the same Actor and Critic models. Before starting with DDPG I tried several MADDPG implementations with separate Actor-Critic, but found that its learning was slow and instable. In most successful MADDPG implementations the environment could solved with about 3000 episodes. Thus, I decided to switch to DDPG which started showing better progress from the beginning. After tuning hyperparameters the environment has been solved with N episodes.

The Actor network hyperparameters:
* Hidden layers: 2
* 1st hidden layer nodes (FC): 256
* 2nd hidden layer nodes (FC): 128
* 3nd hidden layer nodes (FC): 64
* Output layer nodes (actions): 4
* Input parameters (states): 24
* Activation function: ReLU (except output - tanh)
* Batch normalization for input and hidden layers

The Critic network hyperparameters:
* Hidden layers: 2
* 1st hidden layer nodes (FC): 256
* 2nd hidden layer nodes (FC): 128
* 3nd hidden layer nodes (FC): 64
* Output layer nodes (Q-value): 1
* Input parameters (states): 24
* Activation function: ReLU
* Batch normalization for input and 1st hidden layers

### Training
The agent model was trained on AWS (P3.2xlarge).
The following hyperparameters were used:
* Optimizer: Adam
* Replay buffer size: 50000
* Minibatch size: 1024
* Discount factor (Gamma): 0.90
* Learning rate (Actor): 0.001
* Learning rate (Critic): 0.001
* L2 weight decay: 0

The agent was able to solve the environment (0.50+ average scores) in 240 episodes within 10 minutes:

```
Episode 100	Average Score: 0.01	Score: 0.00
Episode 200	Average Score: 0.27	Score: 0.90
Episode 240	Average Score: 0.50	Score: 0.50
Environment solved in 240 episodes!	Average Score: 0.50
Agent training time: 10.6 min
```

![Rewards Plot](plot.png)

### Future Ideas
To improve the agent's performance the following techniques can be used:
- examine MADDPG (separate Actor/Critic) and tune hyperparameters
- change network layers/nodes and choose different hyperparameters
- try other algorithms like PPO, A3C or D4PG
- use prioritised experience buffer
