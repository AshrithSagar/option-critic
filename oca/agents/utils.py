"""
oca/agents/utils.py \n
Utility functions for the agents
"""


def get_agent_meth(meth: str, agent_name: str):
    """Get the agent method based on the agent name."""
    if agent_name == "OptionCritic":
        from .option_critic import evaluate, run
    elif agent_name == "SARSA":
        from .sarsa import run
    elif agent_name == "ACPG":
        from .acpg import run
    else:
        raise ValueError(f"Agent {agent_name} not found.")

    try:
        return {"run": run, "evaluate": evaluate}[meth]
    except KeyError:
        raise ValueError(f"Method {meth} not found for agent {agent_name}.")
