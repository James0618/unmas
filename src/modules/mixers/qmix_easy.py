import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QmixEasy(nn.Module):
    def __init__(self, args):
        super(QmixEasy, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        q_tot = th.sum(agent_qs, dim=-1, keepdim=True) + v
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot
