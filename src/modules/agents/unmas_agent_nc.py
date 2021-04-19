import torch.nn as nn
import torch.nn.functional as F
import torch


class UNMASAgentNC(nn.Module):
    def __init__(self, input_shape, args):
        super(UNMASAgentNC, self).__init__()
        self.args = args

        self.enemy_shape = int((input_shape - args.n_agents - args.n_actions - 5) /
                               (args.n_agents + args.n_actions - 7))

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.q_value_fixed = nn.Linear(args.rnn_hidden_dim, 6)
        self.q_value_enemy = nn.Sequential(
            nn.Linear(self.enemy_shape, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim // 2, 1)
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        t = inputs.shape[0] // self.args.n_agents                           # the timestep of inputs

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        inter_h = self.fc2(h)

        enemies_features = inputs[:, 4: 4 + (self.args.n_actions - 6) * self.enemy_shape].view(
            inputs.shape[0], -1, self.enemy_shape)

        q_enemies = self.q_value_enemy(enemies_features).view(inputs.shape[0], -1)

        q_fixed = self.q_value_fixed(inter_h)                               # the q values of dead, stop and move
        q = torch.cat((q_fixed, q_enemies), dim=-1)                         # the q values of all actions

        return q, h
