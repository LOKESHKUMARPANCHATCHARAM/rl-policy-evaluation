# POLICY EVALUATION

## AIM
The aim is to implement a reinforcement learning agent that can navigate environments from the gym-walk library, which simulates grid-like or walking environments. The agent will learn an optimal policy to reach desired goals based on the given reward structure.

## PROBLEM STATEMENT
The task is to implement and evaluate a reinforcement learning agent in a walking environment using gym. The agent must learn to make decisions that maximize its total reward through trial and error, based on feedback from the environment.

## POLICY EVALUATION FUNCTION
The policy evaluation function aims to compute the value of a given policy by iteratively calculating the expected rewards of following the policy in each state until convergence, allowing for better estimation of state values under the current policy.

## PROGRAM
### pip installation
```python
DEVLOPED BY : LOKESH KUMAR P
REG NO. : 212222240054
pip install git+https://github.com/mimoralea/gym-walk
```
### Import libraries
```
python
import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
```
### Print Functions
```python
#policy
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

#state_value_function
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

#probability_success
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

#mean_return
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
```
### Slippery Walk 5 MDP
```python
env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)

P

init_state

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)

# Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))

#Second policy
pi_2 = lambda s: {
    0:RIGHT, 1:LEFT, 2:LEFT, 3:RIGHT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]

print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state)*100,
    mean_return(env, pi_2)))
```
### Policy Evaluation
```python
import numpy as np

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        prev_V = np.copy(V)
        delta = 0

        for s in range(len(P)):
            v = 0

            a = pi(s)

            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * prev_V[next_state] * (not done))

            V[s] = v

            delta = max(delta, np.abs(prev_V[s] - V[s]))

        if delta < theta:
            break

    return V

V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

V2=policy_evaluation(pi_2,P)
print_state_value_function(V2,P,n_cols=7,prec=5)

V1
V2

#Policy Comparison
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```
## OUTPUT:
### First Policy 
![image](https://github.com/user-attachments/assets/5752b90b-cdde-4df6-80b6-06282c9ab61b)

![image](https://github.com/user-attachments/assets/57998816-a7e5-4b0c-988d-53d211b2ac7e)

### Second Policy

![image](https://github.com/user-attachments/assets/485947b9-e79d-4750-b825-7addb969f56f)

![image](https://github.com/user-attachments/assets/302d208d-33ab-4abc-90af-97bfdd2f1025)

### Comparison

![image](https://github.com/user-attachments/assets/ec3f3df4-317e-4a61-ba6c-e675378de2b4)

## RESULT:
Thus, the reinforcement learning agent successfully learns an optimal policy for navigating the environment, improving its decisions through iterations.
