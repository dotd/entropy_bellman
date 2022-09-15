import numpy as np
from src.utils.StringUtils import StringBuffer

'''
The convention (y, x)
'''


class DictionaryYX:

    def __init__(self):
        self.state2index = dict()
        self.index2state = dict()

    def get_state2index(self, state):
        if state not in self.state2index:
            index = len(self.state2index)
            self.state2index[state] = index
            self.index2state[index] = state
        return self.state2index[state]

    def get_debug(self):
        sb = StringBuffer()
        sb.append("state2index")
        for (y, x) in self.state2index:
            sb.append(f"{(y, x)}->{self.state2index[(y, x)]}")
        sb.append("index2state")
        for index in self.index2state:
            sb.append(f"{index}->{self.index2state[index]}")
        return sb.get_string("\n")


class ActionMap:

    def __init__(self, y_max, x_max):
        # actions
        # 0 = up (Y-1)
        # 1 = right (X+1)
        # 2 = down (Y+1)
        # 3 = left (X-1)
        self.y_max = y_max
        self.x_max = x_max

    def get_next_state(self, state_current, action):
        if action == 0:  # up (Y-1)
            state_next = (max(0, state_current[0]-1), state_current[1])
        elif action == 1:  # right (X+1)
            state_next = (state_current[0], min(self.x_max+1, state_current[1]))
        elif action == 2:
            state_next = (min(self.y_max-1, state_current[0]), state_current[1])
        elif action == 3:
            state_next = (state_current[0], max(0, state_current[1]))
        else:
            state_next = state_current
        return state_next


class MDPGridWorld:

    def __init__(self, y_max, x_max, p_noise, random=np.random.RandomState()):
        # Now we create the probabilities for each action
        # P - transition matrix
        # r - reward
        self.random = random
        self.y_max = y_max
        self.x_max = x_max
        self.num_states = self.y_max * self.x_max
        self.P = np.zeros(shape=(4, self.num_states, self.num_states))
        self.dictionary_xy = DictionaryYX()
        self.action_map = ActionMap(self.y_max, self.x_max)
        self.x = 0
        self.y = 0
        for y in range(self.y_max):
            for x in range(self.x_max):
                #  put the tuple x, y into the dictionary
                index_from = self.dictionary_xy.get_state2index((y, x))
                # checking all directions
                for action in range(4):
                    state_next = self.action_map.get_next_state(state_current=(y, x), action=action)
                    index_to = self.dictionary_xy.get_state2index(state_next)
                    self.P[action, index_from, index_to] = 1-p_noise
        for action in range(4):
            for i in range(self.num_states):
                sum_row = np.sum(self.P[action, i, :])
                self.P[action, i, i] += 1-sum_row

    def get_state_as_index(self):
        return self.dictionary_xy.get_state2index((self.y, self.x))

    def reset(self):
        # We choose a random start state
        self.x = self.random.choice(self.x_max)
        self.y = self.random.choice(self.y_max)
        return self.get_state_as_index()

    def set_state(self, *argv):
        if len(argv) == 1:
            self.y = argv[0][0]
            self.x = argv[0][1]
        else:
            self.y = argv[0]
            self.x = argv[1]

    def step(self, action):
        self.action_map.get_next_state(self.s)

    def get_state_as_str(self, delimiter=","):
        return f"{self.y}{delimiter}{self.x}"

    def debug_str(self):
        sb = StringBuffer()
        sb.append(f"y_max{self.y_max}")
        sb.append(f"x_max{self.x_max}")
        sb.append(f"{self.dictionary_xy.get_debug()}")
        return sb.get_string("\n")

    def debug_str_extended(self):
        sb = StringBuffer()
        sb.append(f"y_max{self.y_max}")
        sb.append(f"x_max{self.x_max}")
        for x in range(self.x_max):
            for y in range(self.y_max):
                self.set_state((y, x))
                sb.append(f"state=({self.y}, {self.x})")
                index = self.dictionary_xy.get_state2index((self.y, self.x))
                tuple_back = self.dictionary_xy.index2state[index]
                sb.append(f"tuple to index={index}")
                sb.append(f"index to tuple={tuple_back}")
                for a in range(4):
                    sb.append()

        return sb.get_string("\n")




