from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn.functional as F
import random
import json

from predictor import MLP, MLPConfig
from alpha_mcts import AlphaMCTS

class DataLoader:
    def __init__(self, args):
        self.states = []
        self.args = args
        with open(self.args['data_path'], 'r') as file:
            for line in tqdm(file, desc="Loading data"): 
                self.states.append(json.loads(line))
        self._batch_idx = 0
        
    def next_batch(self):
        num_games = self.args['num_games']
        batch = self.states[:num_games]
        self.states = self.states[num_games:] + self.states[:num_games]
        return batch

class AlphaZero:
    def __init__(self, model, optimizer, mdp, args):
        self.model = model
        self.optimizer = optimizer
        self.mdp = mdp
        self.args = args
        self.data_loader = DataLoader(args)
        
    def selfPlay(self):
        initial_states = self.data_loader.next_batch()
        histories = []
        targets = [] # [(state, mcts_policy, total_return), ...]

        for state in tqdm(initial_states, desc="Self-Play"):
            mcts = AlphaMCTS(mdp=self.mdp, initial_state=state, args=self.args, model=self.model)
            history = [] #[(state, mcts_policy, action, reward), ...]
            current_state = state
            tqdm.write(str(state))
            for _ in range(self.args['game_length']):
                mcts_policy, mcts_value = mcts.search(temperature=1)
                action = np.random.choice(self.mdp.n_actions, p=mcts_policy)
                next_state, reward, terminated = mcts.step(action)
                history.append((current_state, mcts_policy, action, reward))
                current_state = next_state
                #tqdm.write(str([round(float(p), 4) for p in mcts_policy]))
                tqdm.write(self.mdp._action_names[action])
                tqdm.write(str(current_state))
                if terminated: break
            
            total_return = 0 if terminated else mcts_value # value of terminal state is 0 (no future return)
            for state, mcts_policy, action, reward in reversed(history):
                total_return = reward + self.args['discount'] * total_return
                prepared_state = self.mdp.prepare(state)
                targets.append((prepared_state, mcts_policy, total_return))
                    
            histories.append(history)
        
        return histories, targets
                            
    def train(self, targets):
        random.shuffle(targets)
        batch_size = self.args['batch_size']
        for _ in range(0, len(targets), batch_size):
            batch = targets[:batch_size]
            targets = targets[batch_size:] + targets[:batch_size]
            prepared_states, policy_targets, value_targets = zip(*batch)

            state_inputs = torch.tensor(np.array(prepared_states), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=self.model.device)
            
            out_logits, out_value = self.model(state_inputs)
            out_value = out_value.squeeze(-1)
            
            policy_loss = F.cross_entropy(out_logits, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in tqdm(range(self.args['num_iterations'])):
        
            self.model.eval()
            histories, targets = self.selfPlay()
                
            self.model.train()

            for epoch in tqdm(range(self.args['num_epochs']), desc="Training"):
                self.train(targets)

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter': iteration,
                'args': self.args,
            }

            tqdm.write(f"saving checkpoint to {self.args['out_dir']}")
            torch.save(checkpoint, os.path.join(self.args['out_dir'], f'model_{iteration}.pt'))

#----------------------------------------
# Run AlphaZero
#----------------------------------------
from knot_mdp import KnotMDP
import os

out_dir = 'results/alpha_zero'
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')

device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
tqdm.write(f"using device: {device}")

args = {
    'data_path': 'unknots.txt',
    'num_games': 50,
    'game_length': 20,
    'discount': 0.9,
    'num_iterations': 10,
    'num_epochs': 10,
    'batch_size': 10,
    'N_max': 128,
    'num_sims': 30,
    'C' : 1.414,
    'out_dir': out_dir,
    'n_layers': 4,
    'd_layer': 256,
    'dropout': 0.0,
    'lr': 1e-4,
    'weight_decay': 1e-5,
}

mdp = KnotMDP(args)

model_args = dict(
    n_layers = args['n_layers'], 
    d_layer = args['d_layer'], 
    d_input = args['N_max'],
    n_actions = mdp.n_actions, 
    dropout = args['dropout'], 
    device = device,
    )

config = MLPConfig(**model_args)
model = MLP(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

alpha_zero = AlphaZero(model, optimizer, mdp, args)
alpha_zero.learn()