# individual network settings for each actor + critic pair
# see networkforall for details

from torch.optim import Adam

# add OU noise for exploration
from OUNoise import OUNoise
from networkforgat import ActorNetwork, CriticNetwork
from utilities import hard_update


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class DDPGAgent():
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic,
                 hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2, weight_decay=1.0e-5, device='cuda:0'):
        super(DDPGAgent, self).__init__()

        hidden_gat_dim = 64
        self.actor = ActorNetwork(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = CriticNetwork(in_critic, hidden_gat_dim, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = ActorNetwork(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = CriticNetwork(in_critic, hidden_gat_dim, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)
        self.device = device

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

    def act(self, obs, noise=0.0):
        obs = obs.to(self.device)
        action = self.actor(obs).cpu() + noise * self.noise.noise()
        action = action.clamp(-1, 1)
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(self.device)
        action = self.target_actor(obs).cpu() + noise * self.noise.noise()
        action = action.clamp(-1, 1)
        return action
