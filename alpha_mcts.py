import torch
import numpy as np
import math
from tqdm import trange
import random
from torch import nn
import torch.nn.functional as F

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
                policy = F.softmax(logits, dim=-1).cpu().numpy()
                node.expand(policy)
                
            node.backpropagate(value)

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
