import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import torch

# Initialize environment
env = gym.make("CarRacing-v2", render_mode="rgb_array")
env = Monitor(env)

# Create RL model
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs")

# Train the model
model.learn(total_timesteps=100_000)

# Save trained model
model.save("a2c_carracing")

# Evaluate the model
obs = env.reset()
rewards = []
step_count = 0
max_steps = 1000

while step_count < max_steps:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    step_count += 1
    if done:
        obs = env.reset()

# Save evaluation results
avg_reward = np.mean(rewards)
with open("benchmark_results.txt", "w") as f:
    f.write("A2C Training Complete! Trained for 100,000 timesteps.\n")
    f.write(f"Average Reward over {max_steps} steps: {avg_reward:.2f}\n")

print(f"Training complete. Avg reward over {max_steps} steps: {avg_reward:.2f}")
