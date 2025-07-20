cat > visualize_rewards_fixed.py << 'EOF'
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Use A2C_6 (most recent with data)
log_dir = "./tensorboard_logs/A2C_6"

print(f"Loading TensorBoard logs from: {log_dir}")

# Load TensorBoard logs
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Get the reward data
reward_tag = "rollout/ep_rew_mean"
rewards = event_acc.Scalars(reward_tag)
timesteps = [s.step for s in rewards]
reward_vals = [s.value for s in rewards]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(timesteps, reward_vals, color='green', linewidth=1.5, marker='o', markersize=3)
plt.xlabel("Training Steps")
plt.ylabel("Average Episode Reward")
plt.title(f"A2C Training Progress: {reward_tag}")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Print statistics
print(f"Total data points: {len(reward_vals)}")
print(f"Training steps: {timesteps[0]} to {timesteps[-1]}")
print(f"Initial reward: {reward_vals[0]:.2f}")
print(f"Final reward: {reward_vals[-1]:.2f}")
print(f"Max reward: {max(reward_vals):.2f}")
print(f"Min reward: {min(reward_vals):.2f}")
print(f"Improvement: {reward_vals[-1] - reward_vals[0]:.2f}")

plt.show()
EOF