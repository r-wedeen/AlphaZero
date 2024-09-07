# Vanilla MCTS -- no neural networks and no parallelization
# Assuming environment has a discrete action space

# NOTE: In addition to the standard methods, the Gymnasium.Env
# environment MUST support the following methods:
#   - get_state() -> np.ndarray ; returns the current state of the environment
#   - get_valid_actions() -> list[int] ; returns a list of valid actions
#   - clone() -> Env ; create a new env w/ initial state the current state

import numpy as np
import math

class Node:
    def __init__(self, args, mdp, state=None, parent=None, \
                 action_taken=None, valid_actions=None):

        self.args = args
        self.mdp = mdp
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = {}
        self.valid_actions = valid_actions

        self.state = state
        self.reward = 0
        self.terminal = False

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
            q_value = (child.value_sum / child.visit_count)
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / (1+child.visit_count))
    
    def expand(self):
        for action in self.valid_actions:
            child = Node(self.args, mdp=self.mdp, parent=self, action_taken=action)
            self.children[action] = child
    
    def rollout_eval(self):
        total = 0
        for _ in range(self.args['num_rollouts']):
            total += self.rollout(self.state, self.args['rollout_depth'])

        value = total / self.args['num_rollouts']
        return value
    
    def rollout(self, state, depth):
        rand_action = np.random.choice(self.mdp.get_valid_actions(state))
        next_state, reward, terminal = self.mdp.transition(state, rand_action)
        if depth == 0 or terminal: return reward
        else: return reward + self.args['discount']*self.rollout(next_state, depth - 1)

    def backpropagate(self, value):
        value = self.reward + self.args['discount']*value
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            self.parent.backpropagate(value) 

class MCTS:
    def __init__(self, mdp, initial_state, args):
        self.mdp = mdp 
        self.args = args
        self.root = Node(self.args, mdp=self.mdp, state=initial_state,\
                          valid_actions=self.mdp.get_valid_actions(state))
        
    def search(self):
        for _ in range(self.args['num_sims']):
            
            node = self.root
            
            while not node.is_leaf():
                node = node.select()
            
            if node.parent is not None:
                node.fill()

            value = node.reward
            if not node.terminal:
                value = node.rollout_eval()
                node.expand()
                
            node.backpropagate(value)

        temperature = self.args['temperature']
        policy = np.zeros(self.mdp.n_actions)
        tempered_policy = np.zeros(self.mdp.n_actions)
        q_values = np.zeros(self.mdp.n_actions)
        
        for action, child in self.root.children.items():
            policy[action] = child.visit_count
            tempered_policy[action] = child.visit_count**(1/temperature)
            q_values[action] = child.value_sum / child.visit_count

        policy /= np.sum(policy)        
        tempered_policy /= np.sum(tempered_policy)
        value = policy @ q_values # Bellman equation
        
        return tempered_policy, value
    
    def step(self, action):
        state, reward, terminal = self.mdp.transition(self.root.state, action)
        self.root = self.root.children[action]
        self.root.parent, self.root.action_taken = None, None
        return state, reward, terminal
    
# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------

from knot_mdp import KnotMDP
import json
from tqdm import tqdm

with open('unknots.txt', 'r') as file:
    unknots = []
    for line in tqdm(file):
        unknots.append(json.loads(line))

args = {
        'C': 1.4,
        'discount': 0.99,
        'num_sims': 200,
        'num_rollouts': 10,
        'rollout_depth': 10,
        'temperature': 0.5,
    }

for unknot in unknots[:100]:
    state = np.array(unknot)
    mdp = KnotMDP()

    mcts = MCTS(mdp, initial_state=state, args=args)

    print(state)
    terminal = False
    while not terminal:
        policy, value = mcts.search()
        action = np.random.choice(mdp.n_actions, p=policy)
        print(mdp._action_names[action])
        state, _, terminal = mcts.step(action)
        print(state)
