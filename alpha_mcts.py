import torch
import numpy as np
import math
from tqdm import trange
import random
from torch import nn
import torch.nn.functional as F
from graphviz import Digraph


class Node:
    def __init__(self, args, mdp, state=None, parent=None, \
                 action_taken=None, valid_actions=None, prior=0):

        self.args = args
        self.mdp = mdp
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = {}
        self.valid_actions = valid_actions

        self.state = state
        self.reward = 0
        self.terminal = False

        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0

    def fill(self):
        state, reward, terminal = self.mdp.transition(self.parent.state, self.action_taken)
        self.state = state
        self.reward = reward
        self.terminal = terminal
        self.valid_actions = self.mdp.get_valid_actions(self.state)
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children.values():
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    # TODO: should I be adding all actions in policy's support?
    #       NOTE: This will cause width to potentially explode 
    #            if network predicts nonzero for invalid actions
    def expand(self, policy):
        for action in self.valid_actions:
            prior = policy[action]
            child = Node(self.args, mdp=self.mdp, parent=self, action_taken=action, prior=prior)
            self.children[action] = child

    def backpropagate(self, value):
        value = self.reward + self.args['discount']*value
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            self.parent.backpropagate(value)

class AlphaMCTS:
    def __init__(self, mdp, initial_state, args, model):
        self.mdp = mdp
        self.args = args
        self.model = model
        self.root = Node(self.args, mdp=self.mdp, state=initial_state,\
                          valid_actions=self.mdp.get_valid_actions(initial_state))
    
    @torch.no_grad()
    def search(self, temperature):  
        for _ in range(self.args['num_sims']):
            node = self.root

            while not node.is_leaf():
                node = node.select()
                
            if node.parent is not None:
                node.fill()
            
            value = node.reward
            if not node.terminal:
                prepared_state = torch.tensor(self.mdp.prepare(node.state),\
                                               dtype=torch.float32, device=self.model.device)
                logits, value = self.model(prepared_state)
                value = value.item()
                policy = F.softmax(logits, dim=-1).cpu().numpy()
                node.expand(policy)
                print(f'logits_min: {torch.min(logits)}, logits_max: {torch.max(logits)}')
            
            node.backpropagate(value)
            print(f'value: {value}')
            print(f'reward: {node.reward}')
            
            
            #FOR DEBUGGING
            # Generate and save the visualization
            dot = self.visualize_tree()
            dot.render('mcts_tree', format='png', cleanup=True)
            input("Press Enter to continue...")

        mcts_policy = np.zeros(self.mdp.n_actions)
        tempered_policy = np.zeros(self.mdp.n_actions)
        q_values = np.zeros(self.mdp.n_actions)
        
        for action, child in self.root.children.items():
            mcts_policy[action] = child.visit_count
            tempered_policy[action] = child.visit_count**(1/temperature)
            q_values[action] = child.value_sum / child.visit_count if child.visit_count > 0 else 0

        mcts_policy /= np.sum(mcts_policy)        
        tempered_policy /= np.sum(tempered_policy)
        value = mcts_policy @ q_values # Bellman equation

        return tempered_policy, value
    
    def step(self, action):
        state, reward, terminated = self.mdp.transition(self.root.state, action)
        self.root = self.root.children[action]
        self.root.parent, self.root.action_taken = None, None
        return state, reward, terminated

    def visualize_tree(self, max_depth=10):
        dot = Digraph(comment='MCTS Tree')
        dot.attr(rankdir='TB', size='12000,20000')
        dot.graph_attr['ranksep'] = '2.0'

        def add_nodes_edges(node, parent_id=None, depth=0):
            if depth > max_depth:
                return
            
            if node.visit_count > 0:
                node_id = str(id(node))
                if node.parent is not None:
                    label = f'N: {node.visit_count}\nQ: {(node.value_sum / node.visit_count):.5f}\n UCB: {node.parent.get_ucb(node):.5f}'
                else:
                    label = f'N: {node.visit_count}\nQ: {(node.value_sum / node.visit_count):.5f}'
                dot.node(node_id, label)
                if parent_id:
                    dot.edge(parent_id, node_id, f'{self.mdp._action_names[node.action_taken]}\n{node.prior:.5f}')

            for child in node.children.values():
                add_nodes_edges(child, node_id, depth + 1)

        add_nodes_edges(self.root)
        return dot
    
#------------------------------------------------------------------------------#
# Testing
#------------------------------------------------------------------------------#

from knot_mdp import KnotMDP
import json
from tqdm import tqdm
from predictor import MLP, MLPConfig

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
    'discount': 0.95,
    'num_iterations': 10,
    'num_epochs': 10,
    'batch_size': 10,
    'N_max': 128,
    'num_sims': 200,
    'C' : 1.414,
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

with open('unknots.txt', 'r') as file:
    unknots = []
    for line in tqdm(file):
        unknots.append(json.loads(line))

args = {
        'C': 1.4,
        'discount': 0.95,
        'num_sims': 100,
        'num_rollouts': 10,
        'rollout_depth': 10,
        'temperature': 0.5,
        'N_max' : 128,
    }

for unknot in unknots[:100]:
    state = np.array(unknot)
    mcts = AlphaMCTS(mdp, initial_state=state, args=args, model=model)

    print(state)
    terminal = False
    while not terminal:
        policy, value = mcts.search(temperature=args['temperature'])
        action = np.random.choice(mdp.n_actions, p=policy)
        print(mdp._action_names[action])
        input("Press Enter to continue...")
        state, _, terminal = mcts.step(action)
        print(state)