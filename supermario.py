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
import time 

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
# for step in range(0):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
# env.close()c

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
    def __init__(self, N=100000, batch_size=32):
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
    'batch_size': 32,
    'gamma': 0.99,
    'max_episode_len': 180,
    'min_progress': 17,
    'action_repeats': 6,
    'frames_per_state': 3
}

replay = ExperienceReplay(N=1500, batch_size=params['batch_size'])
Qmodel = Qnetwork()
model2 = copy.deepcopy(Qmodel)
model2.load_state_dict(Qmodel.state_dict())
sync_freq = 50

loss_fn = torch.nn.MSELoss()
learning_rate = 0.00025
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

eps = 1 
losses = []
ep_lengths = []
e_reward = 0.0
episode_length = 0
epochs = 7127431
env.reset()
state1 = prepare_initial_state(env.render('rgb_array'))
state_deque = deque(maxlen=params['frames_per_state'])
last_x_pos = env.env.env._x_position
start_time = time.time()


for i in range(epochs):
    optimizer.zero_grad()
    episode_length += 1
    qval_pred = Qmodel(state1)
    action = int(policy(qval_pred, eps))

    for j in range(params['action_repeats']):
        state2, e_reward_, done, info = env.step(action)
        last_x_pos = info['x_pos']
        if done:
            state1 = reset_env()
            break
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    
    state2 = torch.stack(list(state_deque), dim=1)
    replay.add_memory(state1, action, e_reward, state2, done)
    e_reward = 0

    if episode_length > params['max_episode_len']:
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            done = True
        else:
            last_x_pos = info['x_pos']
    if done:
        ep_lengths.append(info['x_pos'])
        state1 = reset_env()
        last_x_pos = env.env.env._x_position
        episode_length = 0
    else:
        state1 = state2
    
    if i % 100 == 0:
        print('epoch: ', i)

    if len(replay.memory) < params['batch_size']:
        continue

    loss = minibatch_train()
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

   
    if i % 10000 == 0 or i >= epochs - 1:
        print('Saving model...')
        torch.save({
            'Qmodel_dict': Qmodel.state_dict(),
            'model2_dict': model2.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'losses': losses,
            'epoch': i
            }, 'MarioModel.pt')

    if i % sync_freq == 0:
        model2.load_state_dict(Qmodel.state_dict())

    if eps > 0.1:
        eps -= (1/epochs)

# print out time it took to execute
print("--- %s seconds ---" % (time.time() - start_time))

# load saved model
checkpoint = torch.load('MarioModel.pt')
LoadedModel = Qnetwork()
LoadedModel.load_state_dict(checkpoint['Qmodel_dict'])

losses_ = np.array(losses)
plt.figure(figsize=(8,6))
plt.xlim(0, len(losses_))
plt.ylim(0, max(losses_))
plt.plot(losses_, label='Q loss')
plt.legend()
plt.show()


eps = 0.1
done = True
state_deque = deque(maxlen=params['frames_per_state'])
for step in range(5000):
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