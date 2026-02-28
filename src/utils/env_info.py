from tabulate import tabulate
from quadcopter_lift_env_cfg import NUM_DRONES

def print_env_info(env):
    """
    Prints a formatted summary of the environment's spaces,
    observations, actions, and rewards using tabulate.
    
    Args:
        env: The initialized CoopLiftEnv instance.
    """
    print("\n" + "="*60)
    print(f"{'ENVIRONMENT ARCHITECTURE SUMMARY':^60}")
    print("="*60)

    # 1. Shapes and Spaces
    n = env.num_envs
    A = NUM_DRONES
    obs_dim = env._obs_manager.obs_dim
    act_dim = env._action_manager.action_dim

    spaces_data = [
        ["Observation", f"({obs_dim},)", f"({n * A}, {obs_dim})"],
        ["Action", f"({act_dim},)", f"({n * A}, {act_dim})"]
    ]
    print("\n" + tabulate(spaces_data, 
                          headers=["Space", "Per-Agent Shape", "Flattened Wrapper Shape"], 
                          tablefmt="simple_outline"))

    # 2. Observation Terms
    obs_data = [[name, dim] for name, dim in env._obs_manager._TERM_REGISTRY]
    print("\n" + tabulate(obs_data, 
                          headers=["Observation Term", "Dimension"], 
                          tablefmt="simple_outline"))

    # 3. Action Terms
    act_data = [[name, slc.stop - slc.start, desc] for name, slc, desc in env._action_manager._TERM_REGISTRY]
    print("\n" + tabulate(act_data, 
                          headers=["Action Term", "Dimension", "Description"], 
                          tablefmt="simple_outline"))

    # 4. Reward Terms
    rew_terms = list(env._reward_manager._episode_sums.keys())
    # Format into columns if there are many terms (e.g., 2 columns)
    rew_data = [[t] for t in rew_terms]
    print("\n" + tabulate(rew_data, 
                          headers=["Reward Terms"], 
                          tablefmt="simple_outline"))

    # 5. Termination Signals
    term_data = [
        ["terminated", "Task success or physical violation/crash"],
        ["timed_out", "Maximum episode steps reached"]
    ]
    print("\n" + tabulate(term_data, 
                          headers=["Termination Signal", "Description"], 
                          tablefmt="simple_outline"))
    print("="*60 + "\n")
