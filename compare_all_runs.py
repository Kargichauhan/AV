cat > compare_all_runs.py << 'EOF'
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Directories with data
log_dirs = ["./tensorboard_logs/A2C_4", "./tensorboard_logs/A2C_5", "./tensorboard_logs/A2C_6"]
colors = ['red', 'blue', 'green']

plt.figure(figsize=(14, 8))

for i, log_dir in enumerate(log_dirs):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    rewards = event_acc.Scalars("rollout/ep_rew_mean")
    timesteps = [s.step for s in rewards]
    reward_vals = [s.value for s in rewards]
    
    run_name = log_dir.split('/')[-1]
    plt.plot(timesteps, reward_vals, color=colors[i], linewidth=2, 
             label=f'{run_name} (Final: {reward_vals[-1]:.2f})', marker='o', markersize=2)

plt.xlabel("Training Steps")
plt.ylabel("Average Episode Reward")
plt.title("A2C Training Comparison - All Runs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
EOF
