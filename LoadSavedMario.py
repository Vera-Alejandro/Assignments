from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
from skimage.transform import resize
import  numpy as np
import torch
from torch import gather, nn, scatter_add
from torch import optim
import torch.nn.functional as F
from collections import deque
from random import shuffle
import copy

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
done = True

def downscale_obs(obs, new_size=(42,42), to_gray=True):
    if to_gray:
        return resize(obs, new_size, anti_aliasing=True).max(axis=2)
    else: 
        return resize(obs, new_size, anti_aliasing=True)

def prepare_state(state):
    return torch.from_numpy(downscale_obs(state, to_gray=True)).float().unsqueeze(dim=0)


def prepare_multi_state(state1, state2):
    state1 = state1.clone()
    tmp = torch.from_numpy(downscale_obs(state2, to_gray=True)).float()

    state1[0][0] = state1[0][1]
    state1[0][1] = state1[0][2]
    state1[0][2] = tmp
    return state1

def prepare_initial_state(state, N=3):
    state_ = torch.from_numpy(downscale_obs(state, to_gray=True)).float()
    tmp = state_.repeat((N,1,1))
    return tmp.unsqueeze(dim=0)

def policy(qvalues, eps=None):
    if eps is not None:
        if torch.rand(1) < eps:
            return torch.randint(low=0, high=6, size=(1,))
        else: 
            return torch.argmax(qvalues)
    else:
        return torch.multinomial(F.softmax(F.normalize(qvalues), dim=1), num_samples=1)

class ExperienceReplay:
    def __init__(self, N=1000, batch_size=300):
        self.N = N
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0
    
    def add_memory(self, state1, action, reward, state2, done):
        self.counter += 1
        
        if self.counter % 500 == 0:
            self.shuffle_memory()

        if len(self.memory) < self.N:
            self.memory.append((state1, action, reward, state2, done))
        else: 
            rand_index = np.random.randint(0, self.N-1)
            self.memory[rand_index] = (state1, action, reward, state2, done)
    
    def shuffle_memory(self):
        shuffle(self.memory)

    def get_batch(self):
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size

        if len(self.memory) < 1:
            print("Error: No data in memory.")
            return None

        ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        batch = [self.memory[i] for i in ind]
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch], dim=0)
        action_batch = torch.Tensor([x[1] for x in batch]).long()
        reward_batch = torch.Tensor([x[2] for x in batch])
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch], dim=0)
        done_batch = torch.Tensor([x[4] for x in batch])

        return state1_batch, action_batch, reward_batch, state2_batch, done_batch

class Qnetwork(nn.Module):
    def __init__(self):
        super(Qnetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=(3,3), stride=2, padding=1)
        self.linear1 = nn.Linear(288, 100)
        self.linear2 = nn.Linear(100, 7)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))
        y = y.flatten(start_dim=2)
        y = y.view(y.shape[0], -1, 32)
        y = y.flatten(start_dim=1)
        y = F.elu(self.linear1(y))
        y = self.linear2(y)
        return y

params = {
    'batch_size': 450,
    'gamma': 0.6,
    'max_episode_len': 200,
    'min_progress': 16,
    'action_repeats': 6,
    'frames_per_state': 3
}

replay = ExperienceReplay(N=1500, batch_size=params['batch_size'])
Qmodel = Qnetwork()
model2 = copy.deepcopy(Qmodel)
model2.load_state_dict(Qmodel.state_dict())
sync_freq = 50

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = optim.Adam(Qmodel.parameters(), lr=learning_rate)

def reset_env():
    env.reset()
    state1 = prepare_initial_state(env.render('rgb_array'))
    return state1

def minibatch_train():
    state1_batch, action_batch, reward_batch, state2_batch, done_batch = replay.get_batch()
    action_batch = action_batch.view(action_batch.shape[0],1)
    reward_batch = action_batch.view(reward_batch.shape[0],1)
    done_batch = action_batch.view(done_batch.shape[0],1)
    
    qvals = Qmodel(state1_batch)
    with torch.no_grad():
        qtargets_ = model2(state2_batch)

    qtargets = reward_batch.squeeze() + params['gamma']*((1-done_batch.squeeze()) * torch.max(qtargets_, dim=1)[0])
    X = qvals.gather(dim=1, index=action_batch).squeeze()

    return loss_fn(X, qtargets.detach())

    # load saved model
checkpoint = torch.load('1MarioModel.pt')
LoadedModel = Qnetwork()
LoadedModel.load_state_dict(checkpoint['Qmodel_dict'])


eps = 0.1
done = True
state_deque = deque(maxlen=params['frames_per_state'])
for step in range(25000):
    if done:
        env.reset()
        state1 = prepare_initial_state(env.render('rgb_array'))
    # qval_pred = Qmodel(state1)
    qval_pred = LoadedModel(state1)
    action = int(policy(qval_pred, eps))
    state2, reward, done, info = env.step(action)
    state2 = prepare_multi_state(state1, state2)
    state1 = state2
    env.render()

env.close()