import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from tqdm import tqdm

class Environment:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.entities = {
            'plant': 100,
            'consumer': 50,
            'apex_predator': 10
        }
        self.initial_total = sum(self.entities.values())

    def step(self, action):
        self.actions.append(action)

        if action == 0:
            self.entities['consumer'] += 1
        elif action == 1:
            self.entities['consumer'] -= 1
        elif action == 2:
            self.entities['apex_predator'] += 1
        elif action == 3:
            self.entities['apex_predator'] -= 1
        else:
            pass
            # do nothing

        self.entities['plant'] -= int(self.entities['consumer'] * 0.01)
        self.entities['consumer'] -= int(self.entities['apex_predator'] * 0.01)

        self.observations.append(int(sum(self.entities.values())))

        state = int(sum(self.entities.values()))
        reward = self.get_reward()
        done = self.is_done()
        return state, reward, done

    def get_reward(self):
        current_total = sum(self.entities.values())
        if self.initial_total-current_total <= 5:
            return 100
        else:
            return (self.initial_total-current_total)

    def is_done(self):
        if self.entities['plant'] <= 0 or self.entities['consumer'] <= 0 or self.entities['apex_predator'] <= 0:
            return True
        else:
            return False

num_states = 10000
num_actions = 5
Q_table = np.zeros((num_states, num_actions))

alpha = 0.5
gamma = 0.95
epsilon = 0.1
num_episodes = 10000

states = []

def softmax_exploration(Q_values, lambda_, temperature=0.9):
    rho = softmax((Q_values * lambda_))
    return np.random.choice(len(Q_values), p=rho)

for i_episode in tqdm(range(num_episodes)):
    env = Environment()
    state = sum(env.entities.values())
    done = False

    states = []

    while not done:
        # epsilon greedy
        # if np.random.uniform(0, 1) < epsilon:
        #     action = np.random.choice(num_actions)
        # else:
        #     action = np.argmax(Q_table[state])
        action = softmax_exploration(Q_table[state], lambda_=1)

        new_state, reward, done = env.step(action)
        Q_table[state, action] = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state, action])
        state = new_state
        states.append(state)

plt.xlabel("Time Step")
plt.ylabel("Amount of Species")
plt.plot(states)
plt.show()
np.save('q_table.npy', Q_table)
