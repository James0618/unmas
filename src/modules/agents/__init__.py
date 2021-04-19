REGISTRY = {}

from .rnn_agent import RNNAgent
from .unmas_agent import UNMASAgent
from .unmas_agent_nc import UNMASAgentNC

REGISTRY["rnn"] = RNNAgent
REGISTRY['unmas'] = UNMASAgent
REGISTRY['unmas_nc'] = UNMASAgentNC
