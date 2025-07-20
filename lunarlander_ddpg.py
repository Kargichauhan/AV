from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gymnasium as gym

# Create vectorized environment
vec_env = make_vec_env("LunarLanderContinuous-v3", n_envs=4)

# Add noise for exploration
n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize and train the model
model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, verbose=1, tensorboard_log="./tensorboard_logs")
model.learn(total_timesteps=50000)

# Save the trained model
model.save("ddpg_lunarlander")

# Delete model to test saving/loading
del model  

# Load the trained model
model = DDPG.load("ddpg_lunarlander")

# Evaluate the trained model
obs = vec_env.reset()
rewards = []
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = vec_env.step(action)
    rewards.append(np.mean(reward))

# Save results
avg_reward = np.mean(rewards)
with open("benchmark_results.txt", "w") as f:
    f.write(f"DDPG Training Complete! Trained for 50,000 timesteps.\n")
    f.write(f"Average Reward over 1000 steps: {avg_reward}\n")

print(f"Training complete. Avg reward: {avg_reward}")
