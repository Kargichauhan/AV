import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to your TensorBoard logs
log_dir = "./tensorboard_logs/A2C_1"

# Load TensorBoard logs
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# List all available tags
tags = event_acc.Tags()
print("\nAvailable Tags in TensorBoard Log:")
for category, tag_list in tags.items():
    print(f"{category}:")
    for tag in tag_list:
        print(f"  - {tag}")

print("\n" + "="*50)

# Common reward tag names to try (in order of preference)
possible_reward_tags = [
    "rollout/ep_rew_mean",
    "train/ep_rew_mean", 
    "rollout/mean_reward",
    "train/mean_reward",
    "episode_reward",
    "reward"
]

# Find the first available reward tag
reward_tag = None
available_scalars = tags.get("scalars", [])

for tag in possible_reward_tags:
    if tag in available_scalars:
        reward_tag = tag
        print(f"Found reward tag: {reward_tag}")
        break

# If none of the common tags found, look for any tag containing "reward" or "rew"
if reward_tag is None:
    reward_candidates = [tag for tag in available_scalars 
                        if any(keyword in tag.lower() for keyword in ['reward', 'rew', 'return'])]
    
    if reward_candidates:
        print(f"Found potential reward tags: {reward_candidates}")
        reward_tag = reward_candidates[0]  # Use the first one
        print(f"Using: {reward_tag}")
    else:
        print("No reward-related tags found!")
        print("Available scalar tags:")
        for tag in available_scalars:
            print(f"  - {tag}")
        exit(1)

# Plot the rewards
print(f"\nPlotting data for tag: {reward_tag}")
try:
    rewards = event_acc.Scalars(reward_tag)
    timesteps = [s.step for s in rewards]
    reward_vals = [s.value for s in rewards]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, reward_vals, color='green', linewidth=1.5)
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.title(f"Training Progress: {reward_tag}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add some statistics
    print(f"Total data points: {len(reward_vals)}")
    print(f"Final reward: {reward_vals[-1]:.2f}")
    print(f"Max reward: {max(reward_vals):.2f}")
    print(f"Min reward: {min(reward_vals):.2f}")
    
    plt.show()
    
except Exception as e:
    print(f"Error plotting {reward_tag}: {e}")