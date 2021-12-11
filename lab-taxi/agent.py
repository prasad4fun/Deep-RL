import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, eps_decay=0.9):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.001
        self.eps_decay = eps_decay
        self.alpha = 0.1
        self.gamma = 0.8

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # self.epsilon = max(self.epsilon * self.eps_decay, 0.001)
        # if np.random.random() > self.epsilon:
        #     # Select the greedy action
        #     return np.argmax(self.Q[state])
        # else:
        #     # Select randomly an action
        #     return np.random.choice(np.arange(self.nA))
        probs = np.ones(self.nA) * self.epsilon / self.nA
        probs[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        action = np.random.choice(np.arange(self.nA), p=probs)

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #current action value
        current = self.Q[state][action]

        #sarsa max
        #get next state best action value
        # Qsa_next = np.max(self.Q[next_state]) if next_state else 0

        #update current close to Qsa_next
        # target = reward + (self.gamma * Qsa_next)
        # self.Q[state][action] = current + (self.alpha * (target - current))

        # #expected sarsa
        # policy_s = np.ones(self.nA) * self.epsilon / self.nA
        # policy_s[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        #
        # # get next state best action value
        # Qsa_next = np.dot(self.Q[next_state], policy_s)
        #
        # # update current close to Qsa_next
        # target = reward + (self.gamma * Qsa_next)
        # self.Q[state][action] = current + (self.alpha * (target - current))
        #
        # return self.Q

        if not done:
            probs = np.ones(self.nA) * self.epsilon / self.nA
            probs[np.argmax(self.Q[next_state])] = 1 - self.epsilon + (self.epsilon / self.nA)

            expcted_reward = np.sum(np.multiply(probs, self.Q[next_state]))
            self.Q[state][action] += self.alpha * (reward + self.gamma * expcted_reward - self.Q[state][action])

        return self.Q