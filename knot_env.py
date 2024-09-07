
# KnotEnv is an environment with:
#   - states represented by braid words (np array) of length < N_max 
#     in Br_n with n < N_max + 1: 
#       (*) w = [s_1,...,s_m] with s_i in {+-1, ..., +-(n-1)}
#   - actions consisting of 
#       (*) braid1: [s_i, s_{i+1}, s_i, w] -> [[s_{i+1}, s_i, s_{i+1}], w]
#       (*) braid2: [i, j, w] -> [j, i, w] for ||i|-|j|| > 1 or |i| = |j|
#       (*) conjugate(i): w -> [i, w, -i] for i in {1, ..., n-1}
#       (*) stabilize: w -> [w, n]
#       (*) shift_k: [s_1,...,s_m] -> [s_{k+1},...,s_m,s_1,...,s_k]
#       (*) smartcollapse:
#           (*) remove consecutive inverses
#           (*) remove free strands
#           (*) destabilize: [w, n] -> w
#           (*) remove nonconsecutive inverses
#   - reward = -len(word)

import numpy as np
import random
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from dataclasses import dataclass, field

@dataclass
class KnotEnvConfig:
    initial_state: np.ndarray = field(default_factory=lambda: np.array([]))
    N_max: int = 128

class KnotEnv(Env):

    def __init__(self, config: KnotEnvConfig):
        """Initialize the environment"""
        super().__init__()
        # Static variables
        self.config = config
        self.action_names, self.action_methods = self._get_all_actions()
        self.n_actions = len(self.action_methods)
        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(
            low = -config.N_max * np.ones(config.N_max),
            high = config.N_max * np.ones(config.N_max),
            dtype = np.int16
        )
        # Volatile variables
        self.state = config.initial_state
        self.n_strands = 0
        self.valid_actions = []
        self._update()

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        self.state = self.config.initial_state
        self._update()
        return self.state, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # do nothing if action is not admissible from current state
        if action not in self.valid_actions:
            reward = -len(self.state)
        # otherwise, act and update
        else:
            action_method = self.action_methods[action]
            action_method()
            self._update()
            # punish by word length unless it exceeds N_max, then punish by 4*N_max
            reward = -len(self.state) if len(self.state) <= self.config.N_max else -4 * self.config.N_max

        terminated = False
        truncated = False
        if len(self.state) == 0:
            terminated = True
        if len(self.state) >= self.config.N_max:
            truncated = True
    
        return self.state, reward, terminated, truncated, {}
        
    def close(self):
        super().close()
    
    def get_state(self) -> np.array:
        return self.state

    def get_valid_actions(self) -> list[int]:
        return self.valid_actions
    
    def clone(self):
        new_config = KnotEnvConfig(initial_state=self.state, N_max=self.config.N_max)
        new_env = KnotEnv(new_config)
        return new_env
    
    def _update(self):
        """Update volatile variables"""
        if len(self.state) == 0: self.n_strands = 1
        else: self.n_strands = np.max(np.abs(self.state)).astype(int) + 1
        self.valid_actions = self._get_valid_actions()

    def _get_all_actions(self) -> tuple[list[str], list[callable]]:
        action_names = ['smartcollapse', 'stabilize', 'braid1', 'braid2']
        action_names += [f'shift({i})' for i in range(1, self.config.N_max)]
        action_names += [f'conjugate_{i}' for i in range(1, self.config.N_max + 1)]

        action_methods = [self._smartcollapse,  self._stabilize, self._braid1,  self._braid2]
        action_methods += [(lambda i=i: self._shift(i)) for i in range(1, self.config.N_max)]
        action_methods += [(lambda i=i: self._conjugate(i)) for i in range(1, self.config.N_max + 1)]
        
        return action_names, action_methods
    
    def _get_valid_actions(self) -> list[int]:
        # add smartcollapse
        valid_action_names = ['smartcollapse']
        # add stabilization
        if len(self.state) < self.config.N_max:
            valid_action_names.append('stabilize')
        # add braid relations
        if self._is_braidable1():
            valid_action_names.append('braid1')
        if self._is_braidable2():
            valid_action_names.append('braid2')
        # add shift actions
        if len(self.state) > 0:
            valid_action_names += [f'shift({i})' for i in range(1, len(self.state))]
        # add conjugation actions
        if len(self.state) <= self.config.N_max - 2:
            valid_action_names += [f'conjugate_{i}' for i in range(1, self.n_strands)]
        
        valid_actions = [self.action_names.index(action_name) for action_name in valid_action_names]
    
        return valid_actions

    def _is_knot(self):
        link_components = self._get_link_components()
        return (len(link_components) == 1)

    def _is_braidable1(self):
        if (len(self.state) < 3): return False
        if (self.state[0] != self.state[2]): return False
        return (np.abs(self.state[1] - self.state[0]) == 1)
    
    def _is_braidable2(self):
        if (len(self.state) < 2): return False
        diff =  np.abs(np.abs(self.state[0]) - np.abs(self.state[1]))
        return (diff > 1 or diff == 0)
                    
    def _braid1(self):
        assert self._is_braidable1(), "cannot apply braid relation to first 3 letters" 
        self.state[0], self.state[1], self.state[2] = self.state[1], self.state[0], self.state[1]
        self._update()
        
    def _braid2(self):
        assert self._is_braidable2(), "cannot commute first 2 letters of braid word"
        self.state[0], self.state[1] = self.state[1], self.state[0]
        self._update()
        
    
    def _conjugate(self, i):
        assert np.abs(i) < self.n_strands, "conjugation by generator that exceeds number of strands"
        self.state = np.insert(self.state, 0, i)
        self.state = np.append(self.state, -i)
        self._update()
    
    
    def _stabilize(self, sign=1):
        self.state = np.append(self.state, sign*self.n_strands)
        self._update()

    def _destabilize(self):
        if len(self.state) == 0: return
        if np.all(np.abs(self.state[:-1]) < np.abs(self.state[-1])):
            self.state = self.state[:-1] 
            if len(self.state) == 0: self.n_strands = 1
            else: self.n_strands = np.max(np.abs(self.state)).astype(int) + 1
            self._update()
    
    def _shift(self, i):
        self.state = np.roll(self.state, i)
        self._update()

    def _smartcollapse(self):
        word = np.array([])
        while(not np.array_equal(word, self.state)):
            word = self.state
            self._remove_consecutive_inverses()
            self._destabilize()
            self._remove_nonconsecutive_inverses()
            self._remove_free_strands()
        self._update()
    
    def _remove_consecutive_inverses(self):
        if len(self.state) < 2: return
        prev = np.array([])
        while(not np.array_equal(prev, self.state)):
            prev = self.state
            for i in range(len(self.state) - 1):
                if self.state[i] + self.state[i+1] == 0:
                    self.state = np.delete(self.state, [i,i+1])
                    self._update()
                    break

    def _remove_free_strands(self):
        # sort word by absolute value, call it y
        indices = np.argsort(np.abs(self.state))
        y = [self.state[i] for i in indices]

        # permutation to undo sort
        z = {}
        for i in range(len(indices)):
            z[indices[i]] = i
        
        # create a new ordered word, temp, that acts on used strands with free strands deleted and labels shifted down
        # strand 1 < k < n is free if braid word is missing {+-(k-1), +-k}
        # strand k = 1 is free if braid word is missing {+-1} 
        # strand k = n is free if braid word is missing {+-(n-1)}
        count = 1
        temp = []
        for i in range(len(y)):
            sign = np.sign(y[i])
            if i == 0: count = 1
            if np.abs(y[i]) - np.abs(y[i-1]) == 1:
                count += 1
            elif np.abs(y[i]) - np.abs(y[i-1]) > 1:
                count += 2
            temp += [sign * count]

        # undo sort on temp 
        self.state = np.array([temp[z[i]] for i in range(len(self.state))])
        # cast as integer array
        self.state = self.state.astype(int)

        self._update()
    
    def _remove_nonconsecutive_inverses(self):
        if len(self.state) < 2: return
        if (self.state[0] + self.state[-1] == 0): 
            self.state = self.state[1:-1]
            self._update()

    def _random_conjugate(self):
        if self.n_strands == 1: return
        sign = random.choice([1,-1])
        j = random.choice(range(1, self.n_strands))
        self._conjugate(sign*j)

    def _random_shift(self):
        if len(self.state) == 1: return
        i = random.choice(range(len(self.state)))
        self._shift(i)

    def _random_markov_move(self):
        coin = random.choice(['H','T'])
        if coin == 'H' and self.n_strands != 1:
            self._random_conjugate()
        else: 
            sign = random.choice([1,-1])
            self._stabilize(sign)

    def _get_link_components(self):
        # identity permutation
        perm = {i : i for i in range(1, self.n_strands+1)}
        # compute Br_n --> S_n
        for sigma in self.state:
            k = np.abs(sigma).astype(int)
            perm[k], perm[k+1] = perm[k+1], perm[k]
        # find cycles
        cycle_list = []
        ix = 1
        cycle = []
        while perm:
            cycle.append(ix)
            new_ix = perm[ix]
            del perm[ix]
            if new_ix not in perm:
                cycle_list.append(cycle)
                cycle = []
                if perm: ix = min(perm)
            else: ix = new_ix
        return cycle_list

    def _knotify(self):
        link_components = self._get_link_components()
        strands = np.array(range(1, self.n_strands + 1))
        while(len(link_components) > 1):
            break_outer = False
            for component in link_components:
                for strand in component:
                    up, down = strand + 1, strand - 1
                    sign = random.choice([-1,1])
                    if up not in component and up in strands:
                        self.state = np.append(self.state, sign*strand)
                        break_outer = True
                        break
                    elif down not in component and down in strands:
                        self.state = np.append(self.state, sign * (strand-1))
                        break_outer = True
                        break
                if break_outer: break
            link_components = self._get_link_components()
        self._update()