# MAGAT
This is a variation of the MADDPG algorithm enhanced with Graph Attention Layers on the critic networks 

## Simple spread env
This algorithm has been used to solve the simple spread (Cooperative navigation) environment from OpenAI [link](https://github.com/openai/multiagent-particle-envs). N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions. However, I modified part of the reward function to be able to increase the training performance (i.e. the agents receive +10 if they are near a landmark).

<img src="https://github.com/imasmitja/MADDPG-AUV/blob/main/model/episode-49002.gif" width="300" height="500"/>

