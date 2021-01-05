import numpy as np
import torch
from Gridworld import Gridworld
import random 
from matplotlib import pyplot as plt
from collections import deque
import copy

l1 = 64
l2 = 150
l3 = 100
l4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4),
)

model2 = copy.deepcopy(model)
model2.load_state_dict(model.state_dict())
sync_freq = 50

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
epsilon = 1.0

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}


mem_size = 5000
batch_size = 1000
epochs = 2500
max_moves = 50
replay = deque(maxlen=mem_size)
losses = []

j = 0 # frequency count
for i in range(epochs):
    game = Gridworld(4, mode='random')
    state_raw = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state1 = torch.from_numpy(state_raw).float()
    
    status = 1
    moves  = 0
    while(status == 1):
        j += 1
        moves += 1
        qvalue = model(state1)
        qval_np = qvalue.data.numpy()

        if(random.random() < epsilon):
            action_num = np.random.randint(0,4)
        else:
            action_num = np.argmax(qval_np)
        
        action = action_set[action_num]
        
        game.makeMove(action)

        state2_raw = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state2 = torch.from_numpy(state2_raw).float()
        reward = game.reward()
        done = True if reward > 0 else False
        exp = (state1, action_num, reward, state2, done)
        replay.append(exp)
        state1 = state2

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
        
            Q_vals_batch = model(state1_batch)
            
            with torch.no_grad(): 
                Q_target_batch = model2(state2_batch)

            Q_targets = reward_batch + gamma * ((1 - done_batch) * torch.max(Q_target_batch, dim=1)[0])
            X = Q_vals_batch.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            loss = loss_fn(X, Q_targets.detach())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())


        if reward != -1 or moves > max_moves:
            status = 0
            moves = 0

    if epsilon > 0.1:
        epsilon -= ( 1 / epochs)

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

def test_model(model, mode='static', display=True):
    i = 0

    game = Gridworld(4, mode=mode)
    state_raw = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    
    state1 = torch.from_numpy(state_raw).float()

    if display:
        print("Initial State:")
        print(game.display())

    status = 1
    
    while(status == 1):
        qvalue = model(state1)
        qval_np = qvalue.data.numpy()
        
        action_num = np.argmax(qval_np)
        
        action = action_set[action_num]

        if display:
            print('Move #L %s; Taking action %s' % (i, action))
        
        game.makeMove(action)        
        
        state_raw = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state1 = torch.from_numpy(state_raw).float()
        
        if display:
            print(game.display())

        reward = game.reward()

        if reward != -1:
            if reward > 0:
                status = 2 # means we won
                if display:
                    print("Game Won! Reward %s" % (reward,))
            else: 
                status = 0
                if display: 
                    print("Game LOST. Reeward %s" % (reward,))

        i += 1 
        if (i > 15):
            if display: 
                print("Game LOST; too many moves.")
            break
    #end while loop
    win = True if status == 2 else False
    return win 

num_games = 100
numwins = 0
for i in range(num_games ):
    if test_model(model, mode='random', display=False):
        numwins += 1


win_rate = numwins / num_games

