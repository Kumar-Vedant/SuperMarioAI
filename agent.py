import torch
import numpy as np
from neural_net import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Agent:
    def __init__(self, input_dims, num_actions, lr=0.00025, gamma=0.9, epsilon=1.0, eps_decay=0.9999975, eps_min=0.1, replay_buffer_capacity=100_000, batch_size=32, sync_network_rate=1000):

        self.num_actions = num_actions
        self.learn_step_counter = 0

        # set hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # create 2 networks - an online one and a target
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # set optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss()

        # setup replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    # use epsilon-greedy to choose an action
    def choose_action(self, observation):
        # explore
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        # exploit (pick the action with the highest Q-value)
        observation = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    # store a 4-tuple in the replay buffer (state, action, reward, next state) with a "done" flag
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    # update the target network to the current online network
    def sync_networks(self):
        # if a specified no. of learning steps are over
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        # if there aren't enough observations to complete a batch
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()
        # reset gradients to avoid accumulation from previous steps
        self.optimizer.zero_grad()

        # get samples randomly from the replay buffer
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        # get all parameters for each sample
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        # pass the states through the online network
        predicted_q_values = self.online_network(states)
        # pick only the actions that were taken (advanced NumPy indexing)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # pass the next states into the target network and get the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # update the target q_values (Qn = r + yQn-1), (1-done) ensures that rewards for all states after the terminal state is set to 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # calculate loss
        loss = self.loss(predicted_q_values, target_q_values)
        # calculate gradients using backpropagation
        loss.backward()
        # perform gradient descent using those gradients
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()