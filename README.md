# Numerical experiments with no-regret learning algorithms in applications of repeated two-player zero-sum “Markov games”


## Introduction
The vast majority of Reinforcement Learning (RL) problems consists of one agent interacting with an environment. After the agent takes some kind of action, the environment returns a reward, noticing the agent how well the action is. Not to get too deep inside mathmatics, a basic Reinforcement Learning problems seeks a way to gain maximum return out of the environment by taking various actions when encountering different states. When training one single agent, this setting usually works, given the agent maximizing its own reward. But when there are more than one agent, or a multi-agent setting, agents that seek to maximize their own profit may be irrational.

For example, the classic game theory case - Prisoners Dilemma state that the most rational thing to do, is not to gain maximum profit with both prisoners lying, but rather to confess. If training multi-agent RL problems are treated individually, then outcomes would probably reach ones that maximize payoff but irrational. Our goal is to train a multi-agent RL based on a zero-sum Markov game, and try to observe the "rationality" of the agents. We believe that if agents we train are rational, the payoff earned from a zero-sum game would converge to zero, meaning both players would have some strategy to counter the other. 

![](https://policonomics.com/wp-content/uploads/2016/02/Prisoners-dilemma.jpg)


In order to complete our research, we seek help from a research: [Last-iterate Convergence of Decentralized Optimistic Gradient Descent/Ascent in Infinite-horizon Competitive Markov Games](https://arxiv.org/pdf/2102.04540.pdf). This paper provides a decentralized algorithm, meaning that agents do not need to know the full parameter of the other agents and can run independently on each player, requiring only local information such as the player’s own action and the corresponding reward feedback. Compared to other papers that have a centralized algorithm, decentralized ones are more versatile and can potentially run on different environments, no matter it being a cooperative or competitive one. It also makes more sense in the rational dimension, given each player can only response to the others action and not have insight in the other player's policy.

## Algorithm
The main goal of our research is to create a payoff matrix that fits the environment/game. Just like the prisoner's dilemma, we first need to construct a payoff matrix for the environment, then train two agents accordingly. The agents are trained and updated with the payoff matrix so that they choose the best response to the responding matrix. There are a few problems we need to solve when constructing the payoff matrix and the agents. 

#### Constructing Q-function
First of all, the environment is not static. As states shift from one to another, the payoff matrix also changes. To make things easier, imagine a game of blackjack, the payoff matrix changes according to the cards in your hand. The same happens here, the state of the game changes due to the positions of each agent, which then means a different payoff matrix according to the states. This also means there would be a state space amount of matrices with each one being the square of the action space, or in short, a great number. To address this problem, we construct a function Q that takes in the state and actions taken by both players as input, then outputs the payoff of the corresponding actions taken. The Q function is an approximate to the true payoff of the corresponding state and action pair and is updated using Monte-Carlo simulation. The paper gives us insight on how to obtain this approximal.
![](https://i.imgur.com/ybdMEBl.png)

#### Training the Agents
Second, how do we obtain the agents given the Q value. We use policy gradient to update the agent network. Due to zero-sum games, the value for the two agents should be additive inverse, to maximize one means the other side is minimized, which should be prevented according to the other agents policy. As the two agents compete, payoff should gradually converge to zero.


### Main Structure
![](https://i.imgur.com/6QV0c3r.png)

In our implementation, we follow the algorithm provided by the paper roughly, and change some settings to fit our environment. We play T number of games and for each game then update the agents, x and y, using policy gradient as stated in equation 2-5.

In equation 6, the paper suggests a slow update method, using alpha as a weight to control the speed of the update, but in our implementation, we use MSE as a loss to update the value, and does not use the slow update method. Although this brings higher variation to the value, the update is done in a faster pace.

![](https://i.imgur.com/h7pKnk4.png)

Equation 7-8 are approximates that give us the payoff for the corresponding agent. Equation 9 gives us the overall value of the state that is present.

## Experiment Environment 
> ### Derk Gym
<img src="https://i.imgur.com/ltvMpOr.jpg" width="10%"/>
**Figure 1: Derk Gym**

[Derk Gym](http://docs.gym.derkgame.com/) is a multiplayer Online Battle Arena RL-based environment. Two teams of three battle each other while trying to defend their own “statue”. Each team is composed of three units. The goal of the game is trying to attack the opponent's statue and units while defending your own. 

> ### Implementation

<img src="https://i.imgur.com/TofC8Qa.jpg" width="80%"/>
**Figure 2: Network**

In our implementation, we will construct two networks: value network, and policy network. Value network returns “how good” the current state is for a player, we view it as the payoff matrix for a given state. Policy(Actor) network will learn to give an action output by giving a particular input.

<img src="https://i.imgur.com/Nt5Rzwu.jpg" width="80%"/>
**Figure 3: Model Architecture**


We construct our model architecture team based because of our environment setting, see Figure 1. We view the reward, action, and observation from a team perspective.
<img src="https://i.imgur.com/oTYTa3T.jpg" />


Based on the algorithm provided by the paper, our training process can be divided into two parts: Interacting with the environment and Updating Policy and Value Network.

### Interacting with the environment

##### State
When interacting with the environment, we first obtain the observation state. Dimensions for the obersevation include hitpoints of the character, distance and angle to alies and foes, whether an ability is ready, etc, which sums up to have 64 values for each player. We observe a (64,1) state space for each player and hence a (3,64,1) state space for each team.

##### Action
The actions defined by Derk's Gym take five dimensions. The five dimensions being: 1. MoveX. 2. Rotate. 3. ChaseFocus. 4. CastingSlot. 5. ChangeFocus. Hence for every player’s we need to output an action space of (5,1), and each team’s action output will be of dimension (3,5,1). 

When training, we implement epsilon-greedy algorithm to explore the action space. Setting the epsilon to a constant(0.2 in our case), we have a certain chance of choosing a random action to take, and the other 0.8 will follow the policy(the actor model) we've trained. A reward will be recieved immediately after an action is taken.

##### Reward
Rewards in Derk Gym correspond to how well a team is performing and can be set by the user. The reward function we set encourages damage to an enemy unit or damage to the enemy statue. Other fields of reward include time spent away from home territory etc. The environment return each player with its reward individually, we then sum up each teams reward to gain the total reward for each team, then minus the other teams total reward to make the convert the settings to a zero-sum game.

### Updating Policy and Value network 

In the second part, we started to update our policy and value network. We construct a memory buffer to store the action state pair. We use policy gradient to update our policy network. Next, we used NN network to construct our value network. The goal of the network is to approximate the Q-value of each possible state-action pair. We implemented Monte Carlo method in order to better predict our Q-value and update the network based on MSE loss.
 

## Experiment Result
In our testing environment, the parameter settings are: Trajectory(T):3000 for the training process, discount factor（γ）: 0.9 used to relate the rewards to the time domain, exploration rate（ε）: 0.2 for generating exploration step. The optimizer we use to update the network parameter is actor learning rate（η）: 1e-5.

We can see the training process below.

<img src="https://i.imgur.com/R8x8gw0.png" width="80%"/>
During the first 50 training steps, we find that agents do not take good moves, which is reasonable since the agents do not have enough samples and are mostly taking random actions.

<img src="https://i.imgur.com/BAQTsxM.png" width="80%"/>
After training 500 steps, we find out that agents can focus the opponent and group up with its teammates.

<img src="https://i.imgur.com/rmIZDJV.png" width="80%"/>
After 1200 steps, one side found out that grouping up and attacking a statue can result in high rewards, the other team also learned how to defend an attack.

<img src="https://i.imgur.com/RKBjv8A.png" width="80%"/>
Finally after 2500 steps, both teams had part of their team members on attack roles and the others on defence. We can clearly observe that teams seem to cooperate with each other while competing with its opponent.

### Test Result
<img src="https://i.imgur.com/3KZOQ4f.png" width="90%"/>
Based on our assumption that the two competitive agents can pin each other from getting a higher reward. In a zero-sum setting, when two agents compete with each other, the final total reward should be close to 0.

To better compare our result, we set two groups: random agent vs random agent, trained agent vs trained agent. Regarding our testing result, we could find out that the pattern of the trained agent is closer to 0. The experiment met our expectations.

<img src="https://i.imgur.com/QRZOtg2.png" />
Moreover, we tried out different learning rates and discount factor settings. Unfortunately, we couldn't find a specific pattern to describe the best setting.

## Future work and limitation

### Limitation

Due to limited resources, we couldn't extend our training time and try more parameter settings. We may find out more interesting results which would yield the most result from different experiment settings.

### Future work
We may work on experimenting with different algorithms and also test on more environments. In this research, we give more focus on multi-agent competition, we can try to train multi-agent RL with cooperation in the future to observe different results.
