import torch
import numpy as np
from agent_nn import AgentNN
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Agent:
    """
    implement a Deep Q-Network (DQN) for reinforcement learning
    """
    def __init__(self, 
                 input_dims, 
                 num_actions, 
                 lr=0.00025, 
                 gamma=0.9, 
                 epsilon=1.0, 
                 eps_decay=0.99999975, 
                 eps_min=0.1, 
                 replay_buffer_capacity=100_000, 
                 batch_size=32, 
                 sync_network_rate=10000):
        
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # define hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # define networks from agent_nn
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # define optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        # initializ a replay buffer to store experiences
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        """
        implement epsilon-greedy policy for action selection
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()
    

    def decay_epsilon(self):
        """
        decay the epsilon value to reduce exploration over time
        """
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)


    def store_in_memory(self, state, action, reward, next_state, done):
        """
        add experiences to the replay buffer
        """
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32), 
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward), 
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32), 
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))
        

    def sync_networks(self):
        """
        periodically copy the weights from the online network to the target one
        """
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        """
        implement the learning process for DQN
        computing the loss, performing backpropagation, and updating the network weights.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()
        self.optimizer.zero_grad()
        # sample from the replay buffer,
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        # define the keys to extract from the sampled experience
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(states) 
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # calculate the loss between predicted and target Q-values and perform back propagation
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()


        