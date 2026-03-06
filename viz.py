import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import grid2op
from grid2op.PlotGrid import PlotMatplot
from grid2op.Chronics import MultifolderWithCache
from lightsim2grid import LightSimBackend
from config import ENV_NAME, SAVE_PATH, OBS_ATTR_TO_KEEP
from common_utils import create_env

# Set the style for all plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def create_plot_dir():
    """Create directory for saving plots"""
    plot_dir = os.path.join(SAVE_PATH, "dataset_visualizations")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def visualize_grid_topology(env, plot_dir):
    """
    Visualize the grid topology
    """
    print("Visualizing grid topology...")

    try:
        # Initialize the plot helper
        plot_helper = PlotMatplot(env.observation_space)

        # Get an observation
        obs = env.reset()

        # Plot the grid layout
        plt.figure(figsize=(16, 10))
        plot_helper.plot_layout()
        plt.title("Power Grid Network Topology", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grid_topology.png"))
        plt.close()

        # Plot the grid with line details
        plt.figure(figsize=(16, 10))
        # FIXED: changed plot_info to plot_obs as it is more standard, or keep plot_info if intended
        plot_helper.plot_info(observation=obs)
        plt.title("Power Grid Network with Flow Information", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grid_with_flow_info.png"))
        plt.close()

        # Plot the grid with load and generation details
        plt.figure(figsize=(16, 10))
        # FIXED: Changed plot_observation to plot_obs
        plot_helper.plot_obs(observation=obs)
        plt.title("Power Grid Network State", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "grid_state.png"))
        plt.close()

        print("  Grid topology visualizations completed successfully")
    except Exception as e:
        print(f"  Error visualizing grid topology: {str(e)}")
        print("  Skipping grid topology visualizations and continuing with other visualizations")


def analyze_line_statistics(env, plot_dir, n_episodes=10, max_steps=288):
    """
    Analyze statistics of power lines
    """
    print("Analyzing power line statistics...")
    
    # Lists to store data
    rho_data = []
    flow_data = []
    timesteps = []
    episode_numbers = []
    line_names = [f"Line_{i}" for i in range(env.n_line)]
    
    for episode in range(n_episodes):
        obs = env.reset()
        
        for step in range(max_steps):
            # Record line data
            for line_idx in range(env.n_line):
                rho_data.append(obs.rho[line_idx])
                flow_data.append(obs.p_or[line_idx])
                timesteps.append(step)
                episode_numbers.append(episode)
            
            # Take a do-nothing action and observe next state
            obs, _, done, _ = env.step(env.action_space())
            if done:
                break
    
    # Create dataframes
    line_indices = np.tile(np.arange(env.n_line), len(timesteps) // env.n_line)
    line_names_repeated = np.tile(line_names, len(timesteps) // env.n_line)
    
    rho_df = pd.DataFrame({
        'line_idx': line_indices,
        'line_name': line_names_repeated,
        'rho': rho_data,
        'timestep': np.repeat(timesteps[:len(timesteps) // env.n_line], env.n_line),
        'episode': np.repeat(episode_numbers[:len(episode_numbers) // env.n_line], env.n_line)
    })
    
    flow_df = pd.DataFrame({
        'line_idx': line_indices,
        'line_name': line_names_repeated,
        'flow': flow_data,
        'timestep': np.repeat(timesteps[:len(timesteps) // env.n_line], env.n_line),
        'episode': np.repeat(episode_numbers[:len(episode_numbers) // env.n_line], env.n_line)
    })
    
    # Plot line capacity usage (rho) over time
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=rho_df, x='timestep', y='rho', hue='line_idx', alpha=0.7, 
                 palette=sns.color_palette("husl", env.n_line), estimator='mean', 
                 ci='sd', n_boot=100)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Overflow Threshold')
    plt.title("Line Capacity Usage (rho) Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Capacity Usage (rho)")
    plt.legend(title="Line ID", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "line_rho_over_time.png"))
    plt.close()
    
    # Plot line flow distribution
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=flow_df, x='line_idx', y='flow', palette=sns.color_palette("husl", env.n_line))
    plt.title("Power Flow Distribution by Line", fontsize=16)
    plt.xlabel("Line ID")
    plt.ylabel("Power Flow (MW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "line_flow_distribution.png"))
    plt.close()
    
    # Plot heatmap of line capacity usage
    line_rho_means = rho_df.groupby(['line_idx', 'timestep'])['rho'].mean().unstack()
    plt.figure(figsize=(18, 10))
    sns.heatmap(line_rho_means, cmap="YlOrRd", vmin=0, vmax=1.5)
    plt.title("Heatmap of Line Capacity Usage Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Line ID")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "line_capacity_heatmap.png"))
    plt.close()
    
    return rho_df, flow_df

def analyze_load_profile(env, plot_dir, n_episodes=10, max_steps=288):
    """
    Analyze load profiles in the grid
    """
    print("Analyzing load profiles...")
    
    # Lists to store data
    load_p = []
    load_q = []
    timesteps = []
    episode_numbers = []
    load_names = [f"Load_{i}" for i in range(env.n_load)]
    
    for episode in range(n_episodes):
        obs = env.reset()
        
        for step in range(max_steps):
            # Record load data
            for load_idx in range(env.n_load):
                load_p.append(obs.load_p[load_idx])
                load_q.append(obs.load_q[load_idx])
                timesteps.append(step)
                episode_numbers.append(episode)
            
            # Take a do-nothing action and observe next state
            obs, _, done, _ = env.step(env.action_space())
            if done:
                break
    
    # Create dataframes
    load_indices = np.tile(np.arange(env.n_load), len(timesteps) // env.n_load)
    load_names_repeated = np.tile(load_names, len(timesteps) // env.n_load)
    
    load_df = pd.DataFrame({
        'load_idx': load_indices,
        'load_name': load_names_repeated,
        'active_power': load_p,
        'reactive_power': load_q,
        'timestep': np.repeat(timesteps[:len(timesteps) // env.n_load], env.n_load),
        'episode': np.repeat(episode_numbers[:len(episode_numbers) // env.n_load], env.n_load)
    })
    
    # Plot active power demand over time
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=load_df, x='timestep', y='active_power', hue='load_idx', alpha=0.7, 
                 palette=sns.color_palette("muted", env.n_load), estimator='mean', 
                 ci='sd', n_boot=100)
    plt.title("Active Power Demand Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Active Power (MW)")
    plt.legend(title="Load ID", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "load_active_power_over_time.png"))
    plt.close()
    
    # Plot total system load over time
    system_load = load_df.groupby(['timestep', 'episode'])['active_power'].sum().reset_index()
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=system_load, x='timestep', y='active_power', hue='episode', alpha=0.7, 
                 palette=sns.color_palette("deep", n_episodes))
    plt.title("Total System Load Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Total Active Power (MW)")
    plt.legend(title="Episode", loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "total_system_load.png"))
    plt.close()
    
    # Plot load distribution by load bus
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=load_df, x='load_idx', y='active_power', palette=sns.color_palette("muted", env.n_load))
    plt.title("Active Power Distribution by Load", fontsize=16)
    plt.xlabel("Load ID")
    plt.ylabel("Active Power (MW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "load_distribution.png"))
    plt.close()
    
    return load_df

def analyze_generation_profile(env, plot_dir, n_episodes=10, max_steps=288):
    """
    Analyze generation profiles in the grid
    """
    print("Analyzing generation profiles...")
    
    # Lists to store data
    gen_p = []
    gen_v = []
    gen_pmax = []
    gen_pmin = []
    timesteps = []
    episode_numbers = []
    gen_names = [f"Gen_{i}" for i in range(env.n_gen)]
    
    for episode in range(n_episodes):
        obs = env.reset()
        
        for step in range(max_steps):
            # Record generation data
            for gen_idx in range(env.n_gen):
                gen_p.append(obs.prod_p[gen_idx])
                gen_v.append(obs.prod_v[gen_idx])
                gen_pmax.append(env.gen_pmax[gen_idx])
                gen_pmin.append(env.gen_pmin[gen_idx])
                timesteps.append(step)
                episode_numbers.append(episode)
            
            # Take a do-nothing action and observe next state
            obs, _, done, _ = env.step(env.action_space())
            if done:
                break
    
    # Create dataframe
    gen_indices = np.tile(np.arange(env.n_gen), len(timesteps) // env.n_gen)
    gen_names_repeated = np.tile(gen_names, len(timesteps) // env.n_gen)
    
    gen_df = pd.DataFrame({
        'gen_idx': gen_indices,
        'gen_name': gen_names_repeated,
        'active_power': gen_p,
        'voltage': gen_v,
        'pmax': gen_pmax,
        'pmin': gen_pmin,
        'timestep': np.repeat(timesteps[:len(timesteps) // env.n_gen], env.n_gen),
        'episode': np.repeat(episode_numbers[:len(episode_numbers) // env.n_gen], env.n_gen)
    })
    
    # Calculate capacity factor
    gen_df['capacity_factor'] = gen_df['active_power'] / gen_df['pmax']
    
    # Plot active power generation over time
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=gen_df, x='timestep', y='active_power', hue='gen_idx', alpha=0.7, 
                 palette=sns.color_palette("bright", env.n_gen), estimator='mean', 
                 ci='sd', n_boot=100)
    plt.title("Generator Active Power Output Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Active Power (MW)")
    plt.legend(title="Generator ID", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "generator_active_power_over_time.png"))
    plt.close()
    
    # Plot capacity factor over time for each generator
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=gen_df, x='timestep', y='capacity_factor', hue='gen_idx', alpha=0.7, 
                 palette=sns.color_palette("bright", env.n_gen), estimator='mean', 
                 ci='sd', n_boot=100)
    plt.title("Generator Capacity Factor Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Capacity Factor (p/pmax)")
    plt.legend(title="Generator ID", loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "generator_capacity_factor.png"))
    plt.close()
    
    # Plot distribution of generator outputs
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=gen_df, x='gen_idx', y='active_power', palette=sns.color_palette("bright", env.n_gen))
    plt.title("Generator Output Distribution", fontsize=16)
    plt.xlabel("Generator ID")
    plt.ylabel("Active Power (MW)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "generator_output_distribution.png"))
    plt.close()
    
    return gen_df

def plot_supply_demand_balance(load_df, gen_df, plot_dir):
    """
    Plot the balance between supply and demand in the grid
    """
    print("Analyzing supply-demand balance...")
    
    # Aggregate total load and generation for each timestep and episode
    total_load = load_df.groupby(['timestep', 'episode'])['active_power'].sum().reset_index()
    total_load.rename(columns={'active_power': 'total_load'}, inplace=True)
    
    total_gen = gen_df.groupby(['timestep', 'episode'])['active_power'].sum().reset_index()
    total_gen.rename(columns={'active_power': 'total_generation'}, inplace=True)
    
    # Merge dataframes
    balance_df = pd.merge(total_load, total_gen, on=['timestep', 'episode'])
    balance_df['net'] = balance_df['total_generation'] - balance_df['total_load']
    
    # Plot supply-demand balance over time for each episode
    plt.figure(figsize=(16, 8))
    # Plot each episode with low alpha
    episodes = balance_df['episode'].unique()
    for episode in episodes:
        episode_data = balance_df[balance_df['episode'] == episode]
        plt.plot(episode_data['timestep'], episode_data['total_generation'], 
                 color='green', alpha=0.2, linewidth=1)
        plt.plot(episode_data['timestep'], episode_data['total_load'], 
                 color='red', alpha=0.2, linewidth=1)
    
    # Plot the average over all episodes
    avg_balance = balance_df.groupby('timestep').agg({
        'total_generation': 'mean',
        'total_load': 'mean',
        'net': 'mean'
    }).reset_index()
    
    plt.plot(avg_balance['timestep'], avg_balance['total_generation'], 
             color='green', linewidth=2, label='Average Generation')
    plt.plot(avg_balance['timestep'], avg_balance['total_load'], 
             color='red', linewidth=2, label='Average Load')
    
    plt.title("Supply-Demand Balance Over Time", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "supply_demand_balance.png"))
    plt.close()
    
    # Plot the net balance (generation - load)
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=balance_df, x='timestep', y='net', hue='episode', alpha=0.7, 
                 palette=sns.color_palette("deep", len(episodes)), legend=False)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot the average net balance
    plt.plot(avg_balance['timestep'], avg_balance['net'], 
             color='blue', linewidth=3, label='Average Net Balance')
    
    plt.title("Net Power Balance Over Time (Generation - Load)", fontsize=16)
    plt.xlabel("Timestep")
    plt.ylabel("Net Power Balance (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "net_power_balance.png"))
    plt.close()
    
    return balance_df

def visualize_grid_state_changes(env, plot_dir, n_steps=24):
    """
    Visualize how the grid state changes over time
    """
    print("Visualizing grid state changes over time...")

    try:
        # Initialize the plot helper
        plot_helper = PlotMatplot(env.observation_space)

        # Reset environment and get initial observation
        obs = env.reset()

        # Create a directory for the snapshots
        snapshots_dir = os.path.join(plot_dir, "grid_snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)

        # Create grid state snapshots
        for step in range(n_steps):
            # Plot the grid state
            plt.figure(figsize=(16, 10))
            # FIXED: Changed plot_observation to plot_obs
            plot_helper.plot_obs(observation=obs)
            plt.title(f"Grid State at Timestep {step}", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(snapshots_dir, f"grid_state_step_{step:03d}.png"))
            plt.close()

            # Take a do-nothing action
            obs, _, done, _ = env.step(env.action_space())
            if done:
                break

        print(f"  Created {min(step + 1, n_steps)} grid state snapshots")
    except Exception as e:
        print(f"  Error visualizing grid state changes: {str(e)}")
        print("  Skipping grid state change visualizations and continuing with other visualizations")

def analyze_topology_characteristics(env, plot_dir):
    """
    Analyze the topological characteristics of the grid
    """
    print("Analyzing grid topology characteristics...")
    
    # Initialize the plot helper
    plot_helper = PlotMatplot(env.observation_space)
    
    # Get information about the grid
    n_lines = env.n_line
    n_subs = env.n_sub
    n_loads = env.n_load
    n_gens = env.n_gen
    
    # Get line characteristics
    thermal_limits = env.get_thermal_limit()
    line_or_to_subid = env.line_or_to_subid
    line_ex_to_subid = env.line_ex_to_subid
    
    # Create a dataframe with line information
    lines_df = pd.DataFrame({
        'line_idx': np.arange(n_lines),
        'line_name': [f"Line_{i}" for i in range(n_lines)],
        'from_sub': line_or_to_subid,
        'to_sub': line_ex_to_subid,
        'thermal_limit': thermal_limits
    })
    
    # Plot line thermal limits
    plt.figure(figsize=(14, 8))
    bars = plt.bar(lines_df['line_idx'], lines_df['thermal_limit'], color='orangered')
    plt.title("Thermal Limits of Power Lines", fontsize=16)
    plt.xlabel("Line ID")
    plt.ylabel("Thermal Limit (MW)")
    plt.xticks(lines_df['line_idx'])
    plt.grid(axis='y', alpha=0.3)
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}',
                 ha='center', va='bottom', rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "line_thermal_limits.png"))
    plt.close()
    
    # Create network graph representation
    # (This is a simplified version, not using NetworkX to avoid additional dependencies)
    plt.figure(figsize=(16, 12))
    
    # Plot substations as nodes
    sub_coords = {}
    for sub_id in range(n_subs):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        sub_coords[sub_id] = (x, y)
        plt.scatter(x, y, s=300, color='skyblue', edgecolor='navy', zorder=3)
        plt.text(x, y, f"S{sub_id}", fontsize=12, ha='center', va='center', zorder=4)
    
    # Plot lines as edges
    for _, line in lines_df.iterrows():
        from_sub = line['from_sub']
        to_sub = line['to_sub']
        from_coords = sub_coords[from_sub]
        to_coords = sub_coords[to_sub]
        thickness = np.log1p(line['thermal_limit']) / 5  # Scale for better visualization
        plt.plot([from_coords[0], to_coords[0]], [from_coords[1], to_coords[1]], 
                 'grey', linewidth=thickness, alpha=0.7, zorder=1)
    
    plt.title("Grid Network Topology Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "grid_topology_graph.png"))
    plt.close()
    
    return lines_df

def main():
    """
    Main function to generate all visualizations for the Grid2Op dataset
    """
    print(f"Creating visualizations for Grid2Op dataset ({ENV_NAME})...")
    
    try:
        # Create environment
        env = create_env(env_name=ENV_NAME)
        
        # Create plot directory
        plot_dir = create_plot_dir()
        
        # Visualize grid topology
        visualize_grid_topology(env, plot_dir)
        
        try:
            # Analyze line statistics (with shorter episodes for speed)
            rho_df, flow_df = analyze_line_statistics(env, plot_dir, n_episodes=3, max_steps=96)
            
            # Analyze load profiles
            load_df = analyze_load_profile(env, plot_dir, n_episodes=3, max_steps=96)
            
            # Analyze generation profiles
            gen_df = analyze_generation_profile(env, plot_dir, n_episodes=3, max_steps=96)
            
            # Plot supply-demand balance
            balance_df = plot_supply_demand_balance(load_df, gen_df, plot_dir)
        except Exception as e:
            print(f"Error in data collection and analysis: {str(e)}")
            print("Skipping related visualizations")
        
        # Visualize grid state changes (just a few timesteps)
        visualize_grid_state_changes(env, plot_dir, n_steps=12)
        
        try:
            # Analyze topology characteristics
            lines_df = analyze_topology_characteristics(env, plot_dir)
        except Exception as e:
            print(f"Error in topology analysis: {str(e)}")
        
        print(f"All visualizations saved to: {plot_dir}")
        
        # Print some dataset statistics
        print("\nGrid2Op Dataset Statistics:")
        print(f"  Number of substations: {env.n_sub}")
        print(f"  Number of power lines: {env.n_line}")
        print(f"  Number of generators: {env.n_gen}")
        print(f"  Number of loads: {env.n_load}")
        print(f"  Total generation capacity: {env.gen_pmax.sum():.2f} MW")
        print(f"  Average thermal limit: {env.get_thermal_limit().mean():.2f} MW")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        print("Unable to create visualizations. Please check your Grid2Op installation and dataset.")

if __name__ == "__main__":
    main() 