import numpy as np  # Used for array and matrix operations
import random  # Used for random actions and state selection

# ### Class: GridWorld

class GridWorld:
    def __init__(self, size=5):
        """
        **Initializes the GridWorld environment.**
        - The grid has a default size of 5x5.
        - Special points A and B are placed on the grid, each with their corresponding next point and rewards.
        - The Q-table is initialized to zeros, with a 4th dimension for the four possible actions.

        **Parameters:**
        - `size`: The size of the grid (default is 5).
        """
        self.size = size
        self.special = {'A': (0,1), 'B': (0,3)}  # Special points
        self.special_next = {'A': (4,1), 'B': (2,3)}  # Next position for special points
        self.rewards = {'A': 10, 'B': 5}  # Rewards for reaching special points
        self.actions = ['north', 'south', 'east', 'west']  # Possible actions
        self.q_table = np.zeros((size, size, 4))  # Q-table for state-action pairs (size x size grid, 4 actions)

    # ### Function: Step
    def step(self, state, action):
        """
        **Simulates one step in the environment.**
        - The agent moves according to the action unless it reaches special points A or B.

        **Parameters:**
        - `state`: The current position (row, col).
        - `action`: The action to take ('north', 'south', 'east', 'west').

        **Returns:**
        - The new state and the corresponding reward after the action.
        """
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
            return (new_row, new_col), 0  # No reward for regular movement
        else:
            return state, -1  # Penalty for hitting the boundary

    # ### Function: Q-Learning
    def q_learn(self, episodes=5000, gamma=0.9, epsilon=0.1, alpha=0.2):
        """
        **Performs Q-learning to learn the optimal policy.**
        - Runs for a specified number of episodes, updating the Q-table based on agent actions and rewards.
        
        **Parameters:**
        - `episodes`: The number of episodes to run (default is 5000).
        - `gamma`: The discount factor (default is 0.9).
        - `epsilon`: The probability of taking a random action (exploration) (default is 0.1).
        - `alpha`: The learning rate (default is 0.2).
        """
        for _ in range(episodes):
            state = (random.randint(0, self.size-1), random.randint(0, self.size-1))  # Random start state

            for _ in range(100):  # max steps per episode
                # Epsilon-Greedy: Choose random action with probability epsilon, otherwise choose the best action
                if random.random() < epsilon:
                    action_idx = random.randint(0, 3)  # Random action
                else:
                    action_idx = np.argmax(self.q_table[state[0], state[1]])  # Best action from Q-table

                # Perform the action and get the next state and reward
                next_state, reward = self.step(state, self.actions[action_idx])

                # Q-learning update rule
                best_next = np.max(self.q_table[next_state[0], next_state[1]])  # Max Q-value for next state
                self.q_table[state[0], state[1], action_idx] += alpha * (
                    reward + gamma * best_next - self.q_table[state[0], state[1], action_idx])  # Update Q-value

                state = next_state  # Move to the next state

    # ### Function: Get Policy
    def get_policy(self):
        """
        **Extracts the optimal policy and value function from the Q-table.**
        - The optimal policy is determined by choosing the action with the highest Q-value at each state.
        - The value function is the maximum Q-value for each state.

        **Returns:**
        - `policy`: The optimal policy for the agent.
        - `value`: The value function representing the optimal Q-values for each state.
        """
        policy = []
        value = np.zeros((self.size, self.size))  # Initialize the value function

        for i in range(self.size):
            row = []
            for j in range(self.size):
                best_action = np.argmax(self.q_table[i,j])  # Best action for state (i,j)
                row.append(self.actions[best_action])  # Add the corresponding action to the policy
                value[i,j] = np.max(self.q_table[i,j])  # Store the best Q-value (value function)
            policy.append(row)

        return policy, value

# ### Main Program
if __name__ == "__main__":
    print("Training Q-learning agent...")
    gw = GridWorld()  # Initialize the GridWorld environment
    gw.q_learn()  # Train the agent using Q-learning

    policy, value = gw.get_policy()  # Get the optimal policy and value function

    # Print the optimal value function (best achievable reward at each state)
    print("\nOptimal Value Function:")
    for row in value:
        print(" ".join(f"{v:5.2f}" for v in row))  # Format the value for neat output

    # Print the optimal policy (the best action for each state)
    print("\nOptimal Policy:")
    arrows = {'north': '↑', 'south': '↓', 'east': '→', 'west': '←'}  # Arrow symbols for actions
    for row in policy:
        print(" ".join(arrows[action] for action in row))  # Print the arrows corresponding to actions