import numpy as np
import random

class KnotMDP:
    def __init__(self, args):
        self.N_max = args['N_max']
        self._transposition_table = {} # maps (tuple(state), action) to (next_state, reward, terminal)
        self._action_names, self._action_methods = self._get_all_actions()

    def transition(self, state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        if (tuple(state), action) in self._transposition_table:
            return self._transposition_table[(tuple(state), action)]
        next_state = self._action_methods[action](state)
        reward = -float(len(next_state)) / self.N_max
        terminal = (len(next_state) == 0)
        self._transposition_table[(tuple(state), action)] = (next_state, reward, terminal)
        return next_state, reward, terminal
    
    @property
    def n_actions(self) -> int:
        return len(self._action_names)

    def get_valid_actions(self, state: np.ndarray) -> list[int]:
        # add smartcollapse
        valid_action_names = ['smartcollapse']
        # add stabilization
        if len(state) < self.N_max:
            valid_action_names.append('stabilize')
        # add braid relations
        if self._is_braidable1(state):
            valid_action_names.append('braid1')
        if self._is_braidable2(state):
            valid_action_names.append('braid2')
        # add shift actions
        if len(state) > 0:
            valid_action_names += [f'shift({i})' for i in range(1, len(state))]
        # add conjugation actions
        if len(state) <= self.N_max - 2:
            valid_action_names += [f'conjugate_{i}' for i in range(1, self._num_strands(state))]
        
        valid_actions = [self._action_names.index(action_name) for action_name in valid_action_names]
    
        return valid_actions
    
    def prepare(self, state: np.ndarray) -> np.ndarray:
        padded_state = np.pad(state, (0, self.N_max - len(state)), 'constant', constant_values=0)
        return padded_state + self.N_max

    def _get_all_actions(self) -> tuple[list[str], list[callable]]:
        action_names = ['smartcollapse', 'stabilize', 'braid1', 'braid2']
        action_names += [f'shift({i})' for i in range(1, self.N_max)]
        action_names += [f'conjugate_{i}' for i in range(1, self.N_max + 1)]

        action_methods = [self._smartcollapse,  self._stabilize, self._braid1,  self._braid2]
        action_methods += [(lambda state, i=i: self._shift(state, i)) for i in range(1, self.N_max)]
        action_methods += [(lambda state, i=i: self._conjugate(state, i)) for i in range(1, self.N_max + 1)]
        
        return action_names, action_methods
    
    def _is_knot(self, state: np.ndarray) -> bool:
        link_components = self._get_link_components(state)
        return (len(link_components) == 1)
    
    def _num_strands(self, state: np.ndarray) -> int:
        if len(state) == 0: return 1
        return np.max(np.abs(state)).astype(int) + 1

    @staticmethod
    def _is_braidable1(state: np.ndarray) -> bool:
        if (len(state) < 3): return False
        if (state[0] != state[2]): return False
        return (np.abs(state[1] - state[0]) == 1)
    
    @staticmethod
    def _is_braidable2(state: np.ndarray) -> bool:
        if (len(state) < 2): return False
        diff =  np.abs(np.abs(state[0]) - np.abs(state[1]))
        return (diff > 1 or diff == 0)
                    
    def _braid1(self, state: np.ndarray) -> np.ndarray:
        if not self._is_braidable1(state):
            return state
        state[0], state[1], state[2] = state[1], state[0], state[1]
        return state
        
    def _braid2(self, state: np.ndarray) -> np.ndarray:
        if not self._is_braidable2(state):
            return state
        state[0], state[1] = state[1], state[0]
        return state
        
    def _conjugate(self, state: np.ndarray, i: int) -> np.ndarray:
        if np.abs(i) >= self._num_strands(state):
            return state
        state = np.insert(state, 0, i)
        state = np.append(state, -i)
        return state
    
    def _stabilize(self, state: np.ndarray, sign=1) -> np.ndarray:
        state = np.append(state, sign*self._num_strands(state))
        return state

    def _destabilize(self, state: np.ndarray) -> np.ndarray:
        if len(state) == 0: return state
        if np.all(np.abs(state[:-1]) < np.abs(state[-1])):
            state = state[:-1] 
        return state
    
    def _shift(self, state: np.ndarray, i: int) -> np.ndarray:
        state = np.roll(state, i)
        return state

    def _smartcollapse(self, state: np.ndarray) -> np.ndarray:
        word = np.array([])
        while(not np.array_equal(word, state)):
            word = state
            state = self._remove_consecutive_inverses(state)
            state = self._destabilize(state)
            state = self._remove_nonconsecutive_inverses(state)
            state = self._remove_free_strands(state)
        return state
    
    def _remove_consecutive_inverses(self, state: np.ndarray) -> np.ndarray:
        if len(state) < 2: return state
        prev = np.array([])
        while(not np.array_equal(prev, state)):
            prev = state
            for i in range(len(state) - 1):
                if state[i] + state[i+1] == 0:
                    state = np.delete(state, [i,i+1])
                    break
        return state

    def _remove_free_strands(self, state: np.ndarray) -> np.ndarray:
        # sort word by absolute value, call it y
        indices = np.argsort(np.abs(state))
        y = [state[i] for i in indices]

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
        state = np.array([temp[z[i]] for i in range(len(state))])
        # cast as integer array
        state = state.astype(int)

        return state
    
    def _remove_nonconsecutive_inverses(self, state: np.ndarray) -> np.ndarray:
        if len(state) < 2: return state
        if (state[0] + state[-1] == 0): 
            state = state[1:-1]
        return state

    def _random_conjugate(self, state: np.ndarray) -> np.ndarray:
        if self._num_strands(state) == 1: return state
        sign = random.choice([1,-1])
        j = random.choice(range(1, self._num_strands(state)))
        return self._conjugate(state, sign*j)

    def _random_shift(self, state: np.ndarray) -> np.ndarray:
        if len(state) == 1: return
        i = random.choice(range(len(state)))
        return self._shift(state, i)

    def _random_markov_move(self, state: np.ndarray) -> np.ndarray:
        coin = random.choice(['H','T'])
        if coin == 'H' and self._num_strands(state) != 1:
            return self._random_conjugate(state)
        else: 
            sign = random.choice([1,-1])
            return self._stabilize(state, sign)

    def _get_link_components(self, state: np.ndarray) -> list[list[int]]:
        # identity permutation
        perm = {i : i for i in range(1, self._num_strands(state)+1)}
        # compute Br_n --> S_n
        for sigma in state:
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

    def _knotify(self, state: np.ndarray) -> np.ndarray:
        link_components = self._get_link_components(state)
        strands = np.array(range(1, self._num_strands(state) + 1))
        while(len(link_components) > 1):
            break_outer = False
            for component in link_components:
                for strand in component:
                    up, down = strand + 1, strand - 1
                    sign = random.choice([-1,1])
                    if up not in component and up in strands:
                        state = np.append(state, sign*strand)
                        break_outer = True
                        break
                    elif down not in component and down in strands:
                        state = np.append(state, sign * (strand-1))
                        break_outer = True
                        break
                if break_outer: break
            link_components = self._get_link_components(state)
        return state