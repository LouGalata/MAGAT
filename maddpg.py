# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from scipy.spatial import cKDTree

from utilities import soft_update, transpose_to_tensor, transpose_list, gumbel_softmax, register_hooks
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if False else "cpu")

class MADDPG:
    def __init__(self, num_agents=3, discount_factor=0.95, tau=0.02, lr_actor=1.0e-2, lr_critic=1.0e-2,
                 weight_decay=0.0):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        # (in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        # self.maddpg_agent = [DDPGAgent(14, 16, 8, 2, 20, 32, 16),
        #                      DDPGAgent(14, 16, 8, 2, 20, 32, 16),
        #                      DDPGAgent(14, 16, 8, 2, 20, 32, 16)]

        # self.maddpg_agent = [DDPGAgent(18, 64, 32, 2, 24, 64, 32, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay),
        #                      DDPGAgent(18, 64, 32, 2, 24, 64, 32, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay),
        #                      DDPGAgent(18, 64, 32, 2, 24, 64, 32, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay)]
        # layers configuration
        in_actor = num_agents * 2 + (
                    num_agents - 1) * 2 + 2 + 2  # x-y of landmarks + x-y and x-ycoms of others + x-y and x-yvelocity of current agent
        hidden_in_actor = 400
        hidden_out_actor = 200
        out_actor = 2  # each agent have 2 continuous actions on x-y plane
        # in_critic = in_actor * num_agents + out_actor * num_agents  # the critic input is all agents concatenated
        in_critic = in_actor + out_actor
        hidden_in_critic = 700
        hidden_out_critic = 350
        self.maddpg_agent = [
            DDPGAgent(in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic,
                      hidden_out_critic, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay,
                      device=device) for _ in range(num_agents)]

        # self.maddpg_agent = [DDPGAgent(14, 128, 128, 2, 48, 128, 128, lr_actor=lr_actor, lr_critic=lr_critic, weight_decay=weight_decay, device=device) for _ in range(num_agents)]
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

        # initial priority for the experienced replay buffer
        self.priority = 1.

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions_next = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions_next

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions_next = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in
                               zip(self.maddpg_agent, obs_all_agents)]
        return target_actions_next

    def get_adj(self, arr, no_agents=5):
        """
        Take as input the new obs. In position 4 to k, there are the x and y coordinates of each agent
        Make an adjacency matrix, where each agent communicates with the k closest ones
        """
        k_lst = [2, 3]
        points = [i[2:4] for i in arr]
        adj = np.ones((no_agents, no_agents), dtype=float)
        tree = cKDTree(points)
        for cnt, row in enumerate(points):
            dd, ii = tree.query(row, k=k_lst)
            adj[cnt][ii] = 1
        adj = np.fill_diagonal(adj, 0)
        return adj

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents
            Update parameters of agent model based on sample from replay buffer
            Inputs:
                samples: tuple of (observations, full observations, actions, rewards, next
                        observations, full next observations, and episode end masks) sampled randomly from
                        the replay buffer. Each is a list with entries
                        corresponding to each agent
                agent_number (int): index of agent to update
                logger (SummaryWriter from Tensorboard-Pytorch):
                    If passed in, important quantities will be logged
        """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        # obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)

        # import pdb; pdb.set_trace()

        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)

        # adj = [self.get_adj(i) for i in np.swapaxes(np.array([x.numpy() for x in obs]), 1, 0)]
        adj = np.ones((len(obs[0]), len(obs), len(obs)), dtype=float)
        adj = torch.Tensor(adj).to(device)
        obs_full = torch.cat(obs, dim=1)
        next_obs_full = torch.cat(next_obs, dim=1)

         # TODO: review if .to(device) no_agents

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        non_zeros = 0
        for name, param in agent.critic.gat2.named_parameters():
            if 'out_att.W' in name:
                x = param.detach()
                x[x < 0] = 0
                print(x.cpu().numpy())
                non_zeros = np.count_nonzero(x.cpu().numpy())



        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions_next = self.target_act(next_obs)
        target_actions_next_cat = torch.cat(target_actions_next, dim=1)
        # target_critic_input = torch.cat((next_obs_full.t(),target_actions_next), dim=1).to(device)
        target_critic_input = torch.cat((next_obs_full, target_actions_next_cat), dim=1).to(device)

        target_critic_input_gat = torch.cat( (torch.stack(obs), torch.stack(target_actions_next)), dim=2).permute(1,0,2).to(device)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input_gat, adj)

        # Compute Q targets (y) for current states (y_i)

        y = reward[agent_number].view(-1, 1).to(device) + self.discount_factor * q_next * (
                    1 - done[agent_number].view(-1, 1)).to(device)

        # Compute Q expected (q)
        action_cat = torch.cat(action, dim=1)
        # critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        critic_input = torch.cat((obs_full, action_cat), dim=1).to(device)
        stack_obs = torch.stack(next_obs).to(device)
        stack_act = torch.stack(action).to(device)
        critic_input_gat = torch.cat((stack_obs, stack_act), dim=2).permute(1, 0, 2).to(device)
        q = agent.critic(critic_input_gat, adj)  # doing forward(...)

        # Priorized Experience Replay
        # aux = abs(q - y.detach()) + 0.1 #we introduce a fixed small constant number to avoid priorities = 0.
        # aux = np.matrix(aux.detach().numpy())
        # new_priorities = np.sqrt(np.diag(aux*aux.T))

        # import pdb; pdb.set_trace()
        # Compute critic loss
        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(q, y.detach())
        # Compute critic loss
        loss_mse = torch.nn.MSELoss()
        critic_loss = loss_mse(q, y.detach())

        # CHECK IF EXPLODING GRADIENTS IS HAPPENING
        # register_hooks(critic_loss)
        # Minimize the loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        # torch.nn.utils.clip_grad_value_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # update actor network using policy gradient
        # Compute actor loss
        agent.actor_optimizer.zero_grad()
        # make input to agent
        obs_input = obs[agent_number].to(device)
        curr_q_input = self.maddpg_agent[agent_number].actor(obs_input)
        # use Gumbel-Softmax sample
        # curr_q_input = gumbel_softmax(curr_q_input, hard = True) # this should be used only if the action is discrete (for example in comunications, but in general the action is not discrete)
        # detach the other agents to save computation
        # saves some time for computing derivative
        # q_input = [ self.maddpg_agent[i].actor(ob.to(device)) if i == agent_number \
        #            else self.maddpg_agent[i].actor(ob.to(device)).detach()
        #            for i, ob in enumerate(obs) ]
        q_input = [curr_q_input if i == agent_number \
                       else self.maddpg_agent[i].actor(ob.to(device)).detach()
                   for i, ob in enumerate(obs)]

        q_input_cat = torch.cat(q_input, dim=1).to(device)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        # q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        q_input2 = torch.cat((obs_full.to(device), q_input_cat), dim=1)
        obs_st = torch.stack(obs).to(device)
        q_st = torch.stack(q_input).to(device)
        q_input_gat = torch.cat((obs_st, q_st), dim=2).permute(1, 0, 2).to(device)
        actor_loss = -agent.critic(q_input_gat, adj).mean()  # get the policy gradient  # TODO: add the adjacency
        # modification from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py
        actor_loss += (curr_q_input).mean() * 1e-3

        # Minimize the loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al,
                            'non_zeros' : non_zeros
                            },
                           self.iter)

        # return (new_priorities)

    def update_targets(self):
        """soft update targets"""
        # ----------------------- update target networks ----------------------- #
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)







