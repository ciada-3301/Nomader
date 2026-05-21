"""
trainer.py
----------
PPO trainer for Nomader.

Features:
  - Vectorised (multi-env) rollout collection
  - Generalised Advantage Estimation (GAE)
  - PPO clipped surrogate + value loss + entropy bonus
  - Checkpoint save / resume  →  checkpoints/nomader_<step>.pt
  - CSV + console logging
  - Graceful Ctrl-C: saves checkpoint before exit
"""

import os
import sys
import csv
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.nomader_env    import NomaderEnv
from networks.networks  import NomaderPolicy, count_parameters


# ── defaults (can all be overridden via CLI) ─────────────────────────────────
CFG = dict(
    # environment
    num_envs        = 8,          # parallel envs
    # rollout
    steps_per_env   = 256,        # steps collected per env before update
    total_steps     = 2_000_000,  # global training budget
    # PPO hypers
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    vf_coef         = 0.5,
    ent_coef        = 0.01,
    max_grad_norm   = 0.5,
    num_epochs      = 4,          # PPO epochs per rollout
    mini_batch_size = 64,
    # checkpoint
    ckpt_dir        = str(ROOT / "checkpoints"),
    ckpt_every      = 50_000,     # global steps between saves
    resume          = None,       # path to .pt file to resume from
    # misc
    seed            = 42,
    log_every       = 2_000,      # global steps between console prints
)


# ── utilities ────────────────────────────────────────────────────────────────
def obs_to_tensor(obs_list, device):
    """
    Convert list-of-dicts (one per env) → dict-of-batched-tensors.
    """
    keys = obs_list[0].keys()
    out  = {}
    for k in keys:
        arr = np.stack([o[k] for o in obs_list], axis=0)
        out[k] = torch.tensor(arr, device=device)
    return out


def single_obs_to_tensor(obs, device):
    out = {}
    for k, v in obs.items():
        out[k] = torch.tensor(np.expand_dims(v, 0), device=device)
    return out


def latest_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpts    = sorted(ckpt_dir.glob("nomader_*.pt"))
    return str(ckpts[-1]) if ckpts else None


# ── Rollout Buffer ────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, steps, num_envs, device):
        self.T   = steps
        self.N   = num_envs
        self.dev = device
        self.clear()

    def clear(self):
        self.obs_keys    = None
        self.obs_store   = {}
        self.actions     = []
        self.log_probs   = []
        self.rewards     = []
        self.dones       = []
        self.values      = []
        self._ptr        = 0

    def store(self, obs_t, action, log_prob, reward, done, value):
        if self.obs_keys is None:
            self.obs_keys = list(obs_t.keys())
            for k in self.obs_keys:
                self.obs_store[k] = []

        for k in self.obs_keys:
            self.obs_store[k].append(obs_t[k].cpu())

        self.actions.append(action.cpu())
        self.log_probs.append(log_prob.cpu())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.dones.append(torch.tensor(done,   dtype=torch.float32))
        self.values.append(value.cpu())
        self._ptr += 1

    def compute_returns(self, last_value, gamma, gae_lambda):
        """GAE advantage + returns."""
        rewards  = torch.stack(self.rewards)        # (T, N)
        dones    = torch.stack(self.dones)          # (T, N)
        values   = torch.stack(self.values).squeeze(-1)  # (T, N)
        last_v   = last_value.squeeze(-1).cpu()     # (N,)

        advantages = torch.zeros_like(rewards)
        gae        = torch.zeros(self.N)

        for t in reversed(range(self.T)):
            next_val  = last_v if t == self.T-1 else values[t+1]
            next_done = dones[t]
            delta     = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
            gae       = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values

        # flatten (T*N)
        self.flat_obs     = {k: torch.cat(self.obs_store[k], dim=0)
                             for k in self.obs_keys}
        self.flat_actions = torch.cat(self.actions,  dim=0)
        self.flat_logp    = torch.cat(self.log_probs, dim=0)
        self.flat_returns = returns.reshape(-1, 1)
        self.flat_advs    = advantages.reshape(-1, 1)

    def get_minibatches(self, batch_size, device):
        total = self.flat_actions.shape[0]
        idxs  = torch.randperm(total)
        for start in range(0, total, batch_size):
            idx = idxs[start:start+batch_size]
            obs_b = {k: self.flat_obs[k][idx].to(device)
                     for k in self.flat_obs}
            yield (
                obs_b,
                self.flat_actions[idx].to(device),
                self.flat_logp[idx].to(device),
                self.flat_returns[idx].to(device),
                self.flat_advs[idx].to(device),
            )


# ── Trainer ───────────────────────────────────────────────────────────────────
class PPOTrainer:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Nomader] Device: {self.device}")

        # ── envs ──
        seeds = [cfg["seed"] + i for i in range(cfg["num_envs"])]
        self.envs = [NomaderEnv(seed=s) for s in seeds]

        # ── policy ──
        self.policy = NomaderPolicy().to(self.device)
        print(f"[Nomader] Policy parameters: {count_parameters(self.policy):,}")

        self.optimiser = optim.Adam(self.policy.parameters(), lr=cfg["lr"], eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimiser, start_factor=1.0, end_factor=0.1,
            total_iters=cfg["total_steps"] // (cfg["steps_per_env"] * cfg["num_envs"])
        )

        # ── state ──
        self.global_step    = 0
        self.episodes_done  = 0
        self.ep_rewards     = [0.0] * cfg["num_envs"]
        self.ep_lengths     = [0]   * cfg["num_envs"]
        self.completed_rews = []

        # ── logging ──
        Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(cfg["ckpt_dir"]).parent / "logs" / "train.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_init()

        # ── resume ──
        resume = cfg.get("resume") or latest_checkpoint(cfg["ckpt_dir"])
        if resume and Path(resume).exists():
            self._load_checkpoint(resume)
        else:
            print("[Nomader] Starting fresh training.")

        # ── graceful shutdown ──
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        print("\n[Nomader] Ctrl-C caught — saving checkpoint and exiting…")
        self._stop = True

    # ── CSV logging ──────────────────────────────────────────────────────────
    def _csv_init(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["global_step","mean_ep_reward","mean_ep_len",
                            "policy_loss","value_loss","entropy","lr"])

    def _csv_log(self, row):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ── checkpoint I/O ───────────────────────────────────────────────────────
    def _save_checkpoint(self, tag=""):
        step = self.global_step
        name = f"nomader_{step:09d}{tag}.pt"
        path = Path(self.cfg["ckpt_dir"]) / name
        torch.save({
            "global_step"   : step,
            "episodes_done" : self.episodes_done,
            "policy_state"  : self.policy.state_dict(),
            "optim_state"   : self.optimiser.state_dict(),
            "sched_state"   : self.scheduler.state_dict(),
            "cfg"           : self.cfg,
        }, path)
        print(f"[Nomader] ✓ Checkpoint saved → {path.name}")
        return path

    def _load_checkpoint(self, path):
        print(f"[Nomader] Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimiser.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.global_step   = ckpt["global_step"]
        self.episodes_done = ckpt["episodes_done"]
        print(f"[Nomader] Resumed at global_step={self.global_step:,}  "
              f"episodes={self.episodes_done:,}")

    # ── PPO update ───────────────────────────────────────────────────────────
    def _ppo_update(self, buffer):
        cfg = self.cfg
        p_losses, v_losses, entropies = [], [], []

        for _ in range(cfg["num_epochs"]):
            for obs_b, act_b, logp_old, ret_b, adv_b in \
                    buffer.get_minibatches(cfg["mini_batch_size"], self.device):

                _, logp_new, entropy, value = \
                    self.policy.get_action_and_value(obs_b, act_b)

                # normalise advantages within minibatch
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                ratio = (logp_new - logp_old).exp()

                # clipped surrogate
                p1 = ratio * adv_b
                p2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_b
                p_loss = -torch.min(p1, p2).mean()

                # value loss (clipped)
                v_loss = 0.5 * (value - ret_b).pow(2).mean()

                loss = (p_loss
                        + cfg["vf_coef"]  * v_loss
                        - cfg["ent_coef"] * entropy.mean())

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         cfg["max_grad_norm"])
                self.optimiser.step()

                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())

        self.scheduler.step()
        return np.mean(p_losses), np.mean(v_losses), np.mean(entropies)

    # ── main train loop ──────────────────────────────────────────────────────
    def train(self):
        cfg      = self.cfg
        T        = cfg["steps_per_env"]
        N        = cfg["num_envs"]
        ckpt_every = cfg["ckpt_every"]
        next_ckpt  = (self.global_step // ckpt_every + 1) * ckpt_every

        obs_list = [env.reset()[0] for env in self.envs]

        print(f"\n[Nomader] Training for {cfg['total_steps']:,} steps "
              f"across {N} envs  (T={T})\n")

        t_start = time.time()
        last_log_step = self.global_step

        while self.global_step < cfg["total_steps"] and not self._stop:
            buffer = RolloutBuffer(T, N, self.device)

            # ── collect rollout ──────────────────────────────────
            self.policy.eval()
            with torch.no_grad():
                for _ in range(T):
                    obs_t  = obs_to_tensor(obs_list, self.device)
                    action, log_prob, _, value = \
                        self.policy.get_action_and_value(obs_t)

                    acts_np = action.cpu().numpy()
                    rewards, dones = [], []

                    for i, env in enumerate(self.envs):
                        obs_next, r, term, trunc, _ = env.step(acts_np[i])
                        done = term or trunc
                        obs_list[i] = obs_next if not done else env.reset()[0]

                        self.ep_rewards[i] += r
                        self.ep_lengths[i] += 1

                        if done:
                            self.completed_rews.append(self.ep_rewards[i])
                            self.episodes_done += 1
                            self.ep_rewards[i] = 0.0
                            self.ep_lengths[i] = 0

                        rewards.append(r)
                        dones.append(float(done))

                    buffer.store(obs_t, action, log_prob,
                                 np.array(rewards), np.array(dones), value)
                    self.global_step += N

                # bootstrap value for last obs
                last_obs_t = obs_to_tensor(obs_list, self.device)
                last_value = self.policy.get_value(last_obs_t)

            buffer.compute_returns(last_value, cfg["gamma"], cfg["gae_lambda"])

            # ── PPO update ───────────────────────────────────────
            self.policy.train()
            pl, vl, ent = self._ppo_update(buffer)

            # ── logging ─────────────────────────────────────────
            if self.global_step - last_log_step >= cfg["log_every"]:
                last_log_step = self.global_step
                mean_rew = np.mean(self.completed_rews[-50:]) \
                           if self.completed_rews else float("nan")
                elapsed  = time.time() - t_start
                sps      = self.global_step / max(elapsed, 1)
                lr_now   = self.optimiser.param_groups[0]["lr"]

                print(f"step={self.global_step:>8,} | "
                      f"ep={self.episodes_done:>6,} | "
                      f"rew={mean_rew:>7.2f} | "
                      f"p_loss={pl:>6.4f} | v_loss={vl:>6.4f} | "
                      f"ent={ent:>5.3f} | "
                      f"sps={sps:>5.0f} | lr={lr_now:.2e}")

                self._csv_log([self.global_step, mean_rew,
                               np.mean(self.ep_lengths) if self.ep_lengths else 0,
                               pl, vl, ent, lr_now])

            # ── checkpoint ──────────────────────────────────────
            if self.global_step >= next_ckpt:
                self._save_checkpoint()
                next_ckpt += ckpt_every

        # final save
        path = self._save_checkpoint(tag="_final" if not self._stop else "_interrupted")
        print(f"\n[Nomader] Done. {self.global_step:,} steps | "
              f"{self.episodes_done:,} episodes")
        return path


# ── entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Nomader PPO Trainer")
    p.add_argument("--resume",          type=str,   default=None,
                   help="Path to .pt checkpoint to resume from")
    p.add_argument("--total_steps",     type=int,   default=CFG["total_steps"])
    p.add_argument("--num_envs",        type=int,   default=CFG["num_envs"])
    p.add_argument("--steps_per_env",   type=int,   default=CFG["steps_per_env"])
    p.add_argument("--lr",              type=float, default=CFG["lr"])
    p.add_argument("--seed",            type=int,   default=CFG["seed"])
    p.add_argument("--ckpt_every",      type=int,   default=CFG["ckpt_every"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {**CFG, **{k: v for k, v in vars(args).items() if v is not None}}
    trainer = PPOTrainer(cfg)
    trainer.train()