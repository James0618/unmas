import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNMASMixerSum(nn.Module):
    def __init__(self, args, observation_shape):
        super(UNMASMixerSum, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.observation_dim = observation_shape
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed, self.embed_dim))
        self.hyper_w_q = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed, self.embed_dim))
        self.hyper_w_k = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                       nn.ReLU(),
                                       nn.Linear(hypernet_embed, self.embed_dim * 2))

        self.observation_hidden = nn.Sequential(nn.Linear(self.observation_dim, hypernet_embed),
                                                nn.ReLU(),
                                                nn.Linear(hypernet_embed, self.embed_dim))
        self.state_hidden = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                          nn.ReLU(),
                                          nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_q = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        self.hyper_b_k = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, observations, is_alive):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        observation_shape = observations.shape[-1]
        observations = observations.reshape(-1, self.n_agents, observation_shape)

        agent_qs = agent_qs.view(-1, self.n_agents, 1)
        is_alive = is_alive.contiguous().view(-1, self.n_agents, 1)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, 1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # # calculate Q and K separately
        # w_q = th.abs(self.hyper_w_q(states)).view(-1, self.embed_dim, 1)
        # b_q = self.hyper_b_q(states).view(-1, 1, 1)
        # q = th.bmm(hidden, w_q) + b_q
        #
        # state_hidden = self.state_hidden(states)
        # observation_hidden = self.observation_hidden(observations)
        # state_hidden_dup = th.zeros(observation_hidden.shape).to(observation_hidden.device)
        # for i in range(self.n_agents):
        #     state_hidden_dup[:, i, :] = state_hidden
        #
        # weight_hidden = th.cat((observation_hidden, state_hidden_dup), dim=-1)
        # w_k = th.abs(self.hyper_w_k(states)).view(-1, self.embed_dim * 2, 1)
        # b_k = self.hyper_b_k(states).view(-1, 1, 1)
        # k = th.exp(th.bmm(weight_hidden, w_k) + b_k)
        #
        # v = self.V(states).view(-1, 1, 1)
        # q_tot = th.bmm(q.transpose(1, 2), k) / self.n_agents + v
        # q_tot = q_tot.view(bs, -1, 1)

        # calculate Q and K separately
        w_q = th.abs(self.hyper_w_q(states)).view(-1, self.embed_dim, 1)
        b_q = self.hyper_b_q(states).view(-1, 1, 1)
        q = th.bmm(hidden, w_q) + b_q

        state_hidden = self.state_hidden(states)
        observation_hidden = self.observation_hidden(observations)
        state_hidden_dup = th.zeros(observation_hidden.shape).to(observation_hidden.device)
        for i in range(self.n_agents):
            state_hidden_dup[:, i, :] = state_hidden

        weight_hidden = th.cat((observation_hidden, state_hidden_dup), dim=-1)
        w_k = th.abs(self.hyper_w_k(states)).view(-1, self.embed_dim * 2, 1)
        b_k = self.hyper_b_k(states).view(-1, 1, 1)
        k = th.exp(th.bmm(weight_hidden, w_k) + b_k)

        if self.args.weight == 'normal':
            q = q * is_alive
            k = k * is_alive

            k = k / (k.sum(1, keepdim=True) + 0.01)

        v = self.V(states).view(-1, 1, 1)
        q_tot = th.bmm(q.transpose(1, 2), k) + v
        q_tot = q_tot.view(bs, -1, 1)

        return q_tot
