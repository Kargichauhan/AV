import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Hyperparameters
TOTAL_STEPS    = 50_000
ROLLOUT_LENGTH = 1_000
GAMMA          = 0.99
LAMBDA         = 0.95
CLIP_EPS       = 0.2
EPOCHS         = 10
BATCH_SIZE     = 64
LR             = 3e-4
LOG_INTERVAL   = 1_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net     = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU()
        )
        self.mean    = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.net(x)
        return self.mean(x), self.log_std.exp()

    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist      = Normal(mean, std)
        action    = dist.sample()
        logp      = dist.log_prob(action).sum(-1)
        return action, logp, dist


class Value(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae        = 0
    values     = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae   = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages


def main():
    # ─── TensorBoard setup ────────────────────────────────────────────────────
    log_dir = os.path.join(os.getcwd(), "tensorboard_logs", "ppo_scratch")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir, flush_secs=5)
    print("Logging to:", log_dir)

    # ─── Environment & networks ────────────────────────────────────────────────
    env = gym.make("CarRacing-v3", render_mode=None)
    obs_shape = env.observation_space.shape
    obs_dim   = int(np.prod(obs_shape))
    act_dim   = env.action_space.shape[0]

    policy  = Policy(obs_dim, act_dim).to(device)
    value_fn= Value(obs_dim).to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()), lr=LR
    )

    step = 0
    reward_history = []
    print("Starting PPO training...")

    # ─── Main training loop ───────────────────────────────────────────────────
    while step < TOTAL_STEPS:
        rollout_len = min(ROLLOUT_LENGTH, TOTAL_STEPS - step)

        # buffers
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []
        ep_rewards = []

        # reset env
        obs, _ = env.reset()
        obs = torch.tensor(obs / 255.0, dtype=torch.float32)
        obs = obs.permute(2,0,1).flatten().to(device)

        # collect a rollout
        for _ in range(rollout_len):
            obs_buf.append(obs.cpu().numpy())
            with torch.no_grad():
                action, logp, _ = policy.get_action(obs)
                val = value_fn(obs)

            act_buf.append(action.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())
            val_buf.append(val.cpu().numpy())

            next_obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
            done_flag = done or truncated

            rew_buf.append(reward)
            done_buf.append(done_flag)
            ep_rewards.append(reward)

            # prepare next obs
            obs = torch.tensor(next_obs / 255.0, dtype=torch.float32)
            obs = obs.permute(2,0,1).flatten().to(device)
            step += 1

            if done_flag:
                break

        # bootstrap last value
        with torch.no_grad():
            next_val = value_fn(obs).cpu().numpy()
        val_buf.append(next_val)

        # compute GAE advantages and returns
        adv_buf = compute_gae(rew_buf, val_buf, done_buf, GAMMA, LAMBDA)
        ret_buf = [a + v for a, v in zip(adv_buf, val_buf[:-1])]

        # convert all buffers to torch.Tensor (stack obs in one go!)
        obs_buf = torch.from_numpy(
            np.stack(obs_buf, axis=0).astype(np.float32)
        ).to(device)
        act_buf = torch.tensor(act_buf, dtype=torch.float32).to(device)
        logp_buf= torch.tensor(logp_buf, dtype=torch.float32).to(device)
        adv_buf = torch.tensor(adv_buf, dtype=torch.float32).to(device)
        ret_buf = torch.tensor(ret_buf, dtype=torch.float32).to(device)

        # ─── PPO policy/value update ─────────────────────────────────────────
        for _ in range(EPOCHS):
            idx = torch.randperm(len(obs_buf))
            for start in range(0, len(obs_buf), BATCH_SIZE):
                mb_idx = idx[start:start+BATCH_SIZE]
                mb_obs = obs_buf[mb_idx]
                mb_acts= act_buf[mb_idx]
                mb_logp= logp_buf[mb_idx]
                mb_adv = adv_buf[mb_idx]
                mb_ret = ret_buf[mb_idx]

                new_mean, new_std = policy(mb_obs)
                dist             = Normal(new_mean, new_std)
                new_logp         = dist.log_prob(mb_acts).sum(-1)

                ratio     = torch.exp(new_logp - mb_logp)
                clip_adv  = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                p_loss    = -torch.min(ratio*mb_adv, clip_adv).mean()
                v_loss    = (value_fn(mb_obs) - mb_ret).pow(2).mean()
                loss      = p_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ─── logging ─────────────────────────────────────────────────────────────
        avg_reward = float(np.mean(ep_rewards))
        reward_history.append(avg_reward)
        writer.add_scalar("AverageReward", avg_reward, step)
        writer.add_scalar("EpisodeLength", len(ep_rewards), step)

        if step % LOG_INTERVAL < rollout_len:
            print(f"Step: {step}, Avg Reward: {avg_reward:.2f}, Episodes: {len(ep_rewards)}")

    print(f" PPO training complete. Total steps: {step}")
    writer.close()

    # ─── final plot ──────────────────────────────────────────────────────────
    times = np.arange(len(reward_history)) * ROLLOUT_LENGTH
    plt.plot(times, reward_history, label="Avg Reward")
    plt.xlabel("Timesteps"); plt.ylabel("Avg Episode Reward")
    plt.title("PPO on CarRacing-v3")
    plt.grid(True); plt.tight_layout()
    plt.savefig("ppo_training_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
