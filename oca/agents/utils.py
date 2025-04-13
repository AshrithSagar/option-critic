"""
oca/agents/utils.py \n
Utility functions for the agents
"""

from .acpg import run_acpg
from .option_critic import run_oca
from .sarsa import run_sarsa


def get_agent_run(agent_name: str):
    """
    Get the agent run function based on the agent name.
    """
    if agent_name == "OptionCritic":
        return run_oca
    elif agent_name == "SARSA":
        return run_sarsa
    elif agent_name == "ACPG":
        return run_acpg
    else:
        raise ValueError(f"Agent {agent_name} not found.")
