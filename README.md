# AI-Game-Solving-Agent

This repository is for the artificial intelligence game solving agent that I created for the final project of my "Artificial Intelligence" course at The College of New Jersey during the Spring 2020 semester. 

The overall goal was for my agent to be as generalizable to each and any level as possible, and in doing so be able to successfully complete all 10 stages provided for the game. Additionally, while traversing each level the agent needed to try to maximize it's score. 

The method I ended up implementing as the foundation of this agent's
AI was Policy Iteration, which is a Markov Decision Process that deals with partially random environments such as this game with it's moving enemies and varying rewards.

I also needed to add some heuristics to guide the agent to safer paths and actions, but it would be very difficult to be able to achieve the global maximum reward for this game using policy iteration since it would require a specific set of heuristics that nobody knows. I explored the option of using reinforcement learning instead for the agent, but there was not enough time left in the semester to change the entire implementation. 

How the arcade game itself works:

- Each stage has basic fruits, each always worth 100 points

- Each stage also has bonus targets with 3 different outcomes, one can reward 500 points, one can reward 1000 points, and one can reward 0 points and spawn a new enemy

- The stage is completed and our agent moves to the next one when all of the basic fruits are collected. (So the bonus targets are optional)

- The agent starts with 3 lives each game, when the lives run out the game is over

- The agent loses a life when it walks off a high platform or when it touches a enemy or spike

- Each stage has 100 seconds, when that runs out you lose a life

- After completing a stage the agent receives points based on how quickly they completed it

- The game is finished when either all 10 stages are completed or all 3 lives have been used up
