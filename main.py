from envs.env import Environment
from agents.greedy_agent import GreedyAgents
from agents.greedy_agent_optimal import GreedyAgentsOptimal
# from agents.ppo_agent import PPO

import numpy as np
# import argparse

def main():
    # parser = argparse.ArgumentParser(description='Run the delivery simulation')
    # parser.add_argument('--num_agents', type=int, default=5, help='Number of agents')
    # parser.add_argument('--n_packages', type=int, default=10, help='Number of packages')
    # parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of steps')
    # parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    # parser.add_argument('--map', type=str, default="maps/map5.txt", help='Path to map file')
    # args = parser.parse_args()

    print("Please enter the simulation parameters:")
    num_agents = int(input("Enter number of agents (default 5): ") or 5)
    n_packages = int(input("Enter number of packages (default 10): ") or 10)
    max_steps = int(input("Enter maximum steps (default 100): ") or 100)
    seed = int(input("Enter random seed (default 2025): ") or 2025)
    map_file = input("Enter map file path (e.g., maps/map.txt, default maps/map5.txt): ") or "maps/map5.txt"

    print("\nSelect an agent type:")
    print("1: GreedyAgentsOptimal")
    print("2: GreedyAgents")
    # print("3: PPO")
    
    agent_choice = input("Enter agent number (default 1): ") or '1'

    agent_map = {
        '1': GreedyAgentsOptimal,
        '2': GreedyAgents,
        # '3': PPO
    }

    AgentClass = agent_map.get(agent_choice, GreedyAgentsOptimal) 

    # Initialize environment
    env = Environment(
        map_file=map_file,
        max_time_steps=max_steps,
        n_robots=num_agents,
        n_packages=n_packages,
        seed=seed
    )
    state = env.reset()
    env.render(save_frame=True)  # Save initial state

    # Initialize agents
    agents = AgentClass()
    agents.init_agents(state)
    print("Agents initialized.")
    
    # Main simulation loop
    done = False
    while not done:
        actions = agents.get_actions(state)
        state, reward, done, infos = env.step(actions)
        env.render(save_frame=True)  # Save each frame
    
    # Save the simulation as a GIF
    gif_filename = f"simulation_{AgentClass.__name__}_{num_agents}agents_{n_packages}packages.gif"
    gif_path = env.save_gif(gif_filename)
    print(f"\nSimulation completed!")
    print(f"Total reward: {env.total_reward:.2f}")
    print(f"Total time steps: {env.t}")
    print(f"Visualization saved to: {gif_path}")

if __name__ == "__main__":
    main()