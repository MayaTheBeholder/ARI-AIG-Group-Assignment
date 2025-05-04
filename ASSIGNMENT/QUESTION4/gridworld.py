import numpy as np
import random

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.special = {'A': (0,1), 'B': (0,3)}
        self.special_next = {'A': (4,1), 'B': (2,3)}
        self.rewards = {'A': 10, 'B': 5}
        self.actions = ['north', 'south', 'east', 'west']
        self.q_table = np.zeros((size, size, 4))
    
    def step(self, state, action):
        if state in self.special.values():
            if state == self.special['A']:
                return self.special_next['A'], self.rewards['A']
            elif state == self.special['B']:
                return self.special_next['B'], self.rewards['B']
        
        row, col = state
        if action == 'north':
            new_row, new_col = row-1, col
        elif action == 'south':
            new_row, new_col = row+1, col
        elif action == 'east':
            new_row, new_col = row, col+1
        elif action == 'west':
            new_row, new_col = row, col-1
        
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            return (new_row, new_col), 0
        else:
            return state, -1
    
    def q_learn(self, episodes=5000, gamma=0.9, epsilon=0.1, alpha=0.2):
        for _ in range(episodes):
            state = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            
            for _ in range(100):  # max steps per episode
                if random.random() < epsilon:
                    action_idx = random.randint(0, 3)
                else:
                    action_idx = np.argmax(self.q_table[state[0], state[1]])
                
                next_state, reward = self.step(state, self.actions[action_idx])
                
                # Qlearning update
                best_next = np.max(self.q_table[next_state[0], next_state[1]])
                self.q_table[state[0], state[1], action_idx] += alpha * (
                    reward + gamma * best_next - self.q_table[state[0], state[1], action_idx])
                
                state = next_state
    
    def get_policy(self):
        policy = []
        value = np.zeros((self.size, self.size))
        
        for i in range(self.size):
            row = []
            for j in range(self.size):
                best_action = np.argmax(self.q_table[i,j])
                row.append(self.actions[best_action])
                value[i,j] = np.max(self.q_table[i,j])
            policy.append(row)
        
        return policy, value

if __name__ == "__main__":
    print("Training Q-learning agent...")
    gw = GridWorld()
    gw.q_learn()
    
    policy, value = gw.get_policy()
    
    print("\nOptimal Value Function:")
    for row in value:
        print(" ".join(f"{v:5.2f}" for v in row))
    
    print("\nOptimal Policy:")
    arrows = {'north': '↑', 'south': '↓', 'east': '→', 'west': '←'}
    for row in policy:
        print(" ".join(arrows[action] for action in row))