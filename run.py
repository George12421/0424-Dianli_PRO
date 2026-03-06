import os
import time
import subprocess
import argparse

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Grid2Op reinforcement learning algorithms and visualizations')
    parser.add_argument('--visualize-only', action='store_true', help='Run only dataset visualization')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip dataset visualization')
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.visualize_only:
        # Run only dataset visualization
        print("\n" + "="*50)
        print("Visualizing Grid2Op dataset")
        print("="*50)
        subprocess.run(["python", "viz.py"])
    else:
        # Run algorithms
        print("\n" + "="*50)
        print("Running DQN algorithm")
        print("="*50)
        subprocess.run(["python", "dqn.py"])
        
        print("\n" + "="*50)
        print("Running PPO algorithm")
        print("="*50)
        subprocess.run(["python", "ppo.py"])
        
        print("\n" + "="*50)
        print("Running A2C algorithm")
        print("="*50)
        subprocess.run(["python", "a2c.py"])
        
        # Run visualizations if not skipped
        if not args.skip_visualization:
            print("\n" + "="*50)
            print("Visualizing Grid2Op dataset")
            print("="*50)
            subprocess.run(["python", "viz.py"])
    
    # Calculate and print total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"All tasks completed!")
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print("="*50)

if __name__ == "__main__":
    main()
