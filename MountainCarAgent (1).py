import gym
import numpy as np
import math


class MountainCarAgent():
    def __init__(self, buckets=(4, 2), num_episodes=300, min_lr=0.01, min_explore=0.1, discount=0.4, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_explore = min_explore
        self.discount = discount
        self.decay = decay
        self.env = gym.make('MountainCar-v0')
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        
        ### Initializing the Q table only once so that it accumulates knowledge over time 
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.N = np.zeros(self.buckets + (self.env.action_space.n,))

    def get_explore_rate(self, t):
        return max(self.min_explore, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_lr(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def mc_update(self, current_state, new_state, reward, old_action, action,i):
        q = self.Q_table[current_state][old_action]
        q += (1/i)*(reward - q)
        self.Q_table[current_state][old_action] = q
 
    def sarsa_update(self, current_state, new_state, reward, old_action, action,i):
        q = self.Q_table[current_state][old_action]
        q += self.lr*(reward+self.discount*self.Q_table[new_state][action] - q)
        self.Q_table[current_state][old_action] = q

    def ql_update(self, current_state, new_state, reward, old_action, action,i):
        q = self.Q_table[current_state][old_action]
        q += self.lr*(reward+self.discount*np.max(self.Q_table[new_state]) - q)
        self.Q_table[current_state][old_action] = q
            
    def expected_sarsa_update(self, current_state, new_state, reward, old_action, action, i):
        q = self.Q_table[current_state][old_action]
        
        # Create a policy using the current Q-table
        policy = np.ones(self.env.action_space.n) * self.explore_rate / self.env.action_space.n
        
        best_action = np.argmax(self.Q_table[new_state])
        
        policy[best_action] += (1.0 - self.explore_rate)
        # Calculate the expected value
        
        expected_value = np.sum(policy * self.Q_table[new_state])
        # Calculate the new Q-value
        
        q += self.lr*(reward + self.discount * expected_value - q)
        self.Q_table[current_state][old_action] = q
        
        

    def off_policy_expected_sarsa_update(self, current_state, new_state, reward, old_action, i):
        q = self.Q_table[current_state][old_action]
        
        # Create a greedy policy using the current Q-table
        policy = np.zeros(self.env.action_space.n)
        
        best_action = np.argmax(self.Q_table[new_state])
        
        policy[best_action] = 1.0
        # Calculate the expected value
        expected_value = np.sum(policy * self.Q_table[new_state])
        
        # Calculate the new Q-value
        q += self.lr * (reward + self.discount * expected_value - q)
        self.Q_table[current_state][old_action] = q


        
    def choose_action(self, state):
        x = (np.random.uniform(0, 1))
        if  x < self.explore_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])
        

    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(np.round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def train(self, method='mc'):
        losses=[]
        for i, e in enumerate(range(self.num_episodes)):
            i+=1
#             print("Start")/
            total_R=0
            ### Reinitializing the states at the beginning of each episode 
            current_state = self.discretize_state(self.env.reset(options={(-0.6,-0.4), 0})[0])
            self.lr = self.get_lr(e)
            self.explore_rate = self.get_explore_rate(e)
            termintated, truncated, position, end = False,False,False,False
            old_action = 1
            steps=0
            while not any([termintated, truncated, position, end]):
                steps+=1
                end = steps==200
                action = self.choose_action(current_state)
                obs, reward, termintated, truncated, _ = self.env.step(action)
                position = obs[0]>=0.5
                new_state = self.discretize_state(obs)
                
                total_R+=reward

                if method == 'mc':
                    self.mc_update(current_state, new_state, total_R, old_action, action,i)
                if method == 'sarsa':
                    self.sarsa_update(current_state, new_state, total_R, old_action, action,i)
                if method == 'ql':
                    self.ql_update(current_state, new_state, total_R, old_action, action,i)
                    
                if method == 'expected_sarsa':
                    self.expected_sarsa_update(current_state, new_state, total_R, old_action, action, i)
                
                if method == 'off_policy_expected_sarsa':
                    self.off_policy_expected_sarsa_update(current_state, new_state, total_R, old_action, i)
                    
                current_state = new_state
                old_action = action
            losses.append(total_R)
            if position == True:
                print('At episode: ', e, ', Win!!!', sep='')
#             self.mc_update(current_state, new_state, total_R, old_action, action,i)

        print('Finished training!')
        return losses

    def run(self):
       
        self.env = gym.make('MountainCar-v0', render_mode='human')
        current_state = self.discretize_state(self.env.reset(options={(-0.6,-0.4), 0})[0])
        steps=0
        termintated, truncated, position, end = False,False,False,False
        while not any([termintated, truncated, position, end]):
            steps+=1
            end = steps==200
            action = self.choose_action(current_state)
#             print(action)
            obs, reward, termintated, truncated, _ = self.env.step(action)
            position = obs[0]>=0.5
            current_state = self.discretize_state(obs)
        if position == True:
            print('Win!!!')
            
        self.env.close()
