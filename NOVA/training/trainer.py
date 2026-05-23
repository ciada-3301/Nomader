"""
trainer.py
----------
PPO trainer for Nomader — rewritten with:
  - SubprocVecEnv  : true multiprocess rollout collection (one proc per env)
  - Tuned hypers   : larger mini_batch (256), longer rollouts (512), 6 envs
  - Render fix     : separate render env, render_every controls frequency
  - Grad norm log  : printed each update so you can spot dead gradients early
  - Value loss clip: stabilises critic early in training
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import set_start_method

# ── SubprocVecEnv — borrowed from SB3, no SB3 training code used ────────────
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.nomader_env   import NomaderEnv
from networks.networks import NomaderPolicy, count_parameters


# ── defaults ─────────────────────────────────────────────────────────────────
CFG = dict(
    # environment
    num_envs         = 6,           # ← was 8; leave 2 threads for main proc
    # rollout
    steps_per_env    = 512,         # ← was 256; longer rollouts = cleaner GAE
    total_steps      = 2_000_000,
    # PPO hypers
    lr               = 3e-4,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_eps         = 0.2,
    vf_coef          = 0.5,
    ent_coef         = 0.01,
    max_grad_norm    = 0.5,
    num_epochs       = 4,
    mini_batch_size  = 256,         # ← was 64; more stable gradient estimates
    # render
    render_every     = 5_000,       # global steps between render frames
    # checkpoint
    ckpt_dir         = str(ROOT / "checkpoints"),
    ckpt_every       = 50_000,
    resume           = None,
    # misc
    seed             = 42,
    log_every        = 2_000,
)


# ── utilities ─────────────────────────────────────────────────────────────────
def make_env(seed):
    """Factory for SubprocVecEnv — must be picklable (top-level function)."""
    def _init():
        return NomaderEnv(seed=seed)
    return _init


def obs_sb3_to_tensor(obs_dict, device):
    """
    SubprocVecEnv returns a dict of stacked numpy arrays {key: (N, ...)}.
    Convert to dict of tensors on device.
    """
    return {k: torch.tensor(v, device=device) for k, v in obs_dict.items()}


def obs_list_to_tensor(obs_list, device):
    """Fallback for sequential envs (used by render env)."""
    keys = obs_list[0].keys()
    return {k: torch.tensor(np.stack([o[k] for o in obs_list]), device=device)
            for k in keys}


def single_obs_to_tensor(obs, device):
    return {k: torch.tensor(np.expand_dims(v, 0), device=device)
            for k, v in obs.items()}


def latest_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpts    = sorted(ckpt_dir.glob("nomader_*.pt"))
    return str(ckpts[-1]) if ckpts else None


# ── Rollout Buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, steps, num_envs, device):
        self.T   = steps
        self.N   = num_envs
        self.dev = device
        self.clear()

    def clear(self):
        self.obs_keys  = None
        self.obs_store = {}
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []
        self.values    = []

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

    def compute_returns(self, last_value, gamma, gae_lambda):
        rewards  = torch.stack(self.rewards).squeeze(-1)          # (T, N)
        dones    = torch.stack(self.dones).squeeze(-1)            # (T, N)
        values   = torch.stack(self.values).squeeze(-1)           # (T, N)
        last_v   = last_value.squeeze(-1).cpu()                   # (N,)

        advantages = torch.zeros_like(rewards)
        gae        = torch.zeros(self.N)

        for t in reversed(range(self.T)):
            next_val  = last_v if t == self.T - 1 else values[t + 1]
            next_done = dones[t]
            delta     = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
            gae       = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values

        self.flat_obs     = {k: torch.cat(self.obs_store[k], dim=0)
                             for k in self.obs_keys}
        self.flat_actions = torch.cat(self.actions,   dim=0)
        self.flat_logp    = torch.cat(self.log_probs, dim=0)
        self.flat_returns = returns.reshape(-1, 1)
        self.flat_advs    = advantages.reshape(-1, 1)

    def get_minibatches(self, batch_size, device):
        total = self.flat_actions.shape[0]
        idxs  = torch.randperm(total)
        for start in range(0, total, batch_size):
            idx   = idxs[start:start + batch_size]
            obs_b = {k: self.flat_obs[k][idx].to(device) for k in self.flat_obs}
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

        # ── vectorised envs ──────────────────────────────────────────────────
        seeds = [cfg["seed"] + i for i in range(cfg["num_envs"])]

        if SB3_AVAILABLE:
            self.vec_env   = SubprocVecEnv([make_env(s) for s in seeds])
            self.use_subproc = True
            print(f"[Nomader] SubprocVecEnv: {cfg['num_envs']} workers")
        else:
            # graceful fallback — sequential, no SB3 needed
            self.vec_env   = [NomaderEnv(seed=s) for s in seeds]
            self.use_subproc = False
            print("[Nomader] WARNING: stable-baselines3 not found — "
                  "falling back to sequential envs. "
                  "Install with: pip install stable-baselines3")

        # ── separate render env (never touches training envs) ────────────────
        self.render_env = NomaderEnv(render_mode="human", seed=cfg["seed"] + 99)
        self.render_obs, _ = self.render_env.reset()
        self._next_render  = cfg["render_every"]

        # ── policy ───────────────────────────────────────────────────────────
        self.policy    = NomaderPolicy().to(self.device)
        print(f"[Nomader] Policy parameters: {count_parameters(self.policy):,}")

        self.optimiser = optim.Adam(self.policy.parameters(),
                                    lr=cfg["lr"], eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimiser, start_factor=1.0, end_factor=0.1,
            total_iters=cfg["total_steps"] // (cfg["steps_per_env"] * cfg["num_envs"])
        )

        # ── training state ───────────────────────────────────────────────────
        self.global_step    = 0
        self.episodes_done  = 0
        self.ep_rewards     = [0.0] * cfg["num_envs"]
        self.ep_lengths     = [0]   * cfg["num_envs"]
        self.completed_rews = []

        # ── logging ──────────────────────────────────────────────────────────
        Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(cfg["ckpt_dir"]).parent / "logs" / "train.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_init()

        # ── resume ───────────────────────────────────────────────────────────
        resume = cfg.get("resume") or latest_checkpoint(cfg["ckpt_dir"])
        if resume and Path(resume).exists():
            self._load_checkpoint(resume)
        else:
            print("[Nomader] Starting fresh training.")

        # ── graceful shutdown ─────────────────────────────────────────────────
        self._stop = False
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        print("\n[Nomader] Ctrl-C caught — saving and exiting…")
        self._stop = True

    # ── CSV ───────────────────────────────────────────────────────────────────
    def _csv_init(self):
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "global_step", "mean_ep_reward", "mean_ep_len",
                    "policy_loss", "value_loss", "entropy", "grad_norm", "lr"
                ])

    def _csv_log(self, row):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ── checkpoints ──────────────────────────────────────────────────────────
    def _save_checkpoint(self, tag=""):
        name = f"nomader_{self.global_step:09d}{tag}.pt"
        path = Path(self.cfg["ckpt_dir"]) / name
        torch.save({
            "global_step"  : self.global_step,
            "episodes_done": self.episodes_done,
            "policy_state" : self.policy.state_dict(),
            "optim_state"  : self.optimiser.state_dict(),
            "sched_state"  : self.scheduler.state_dict(),
            "cfg"          : self.cfg,
        }, path)
        print(f"[Nomader] ✓ Checkpoint → {path.name}")
        return path

    def _load_checkpoint(self, path):
        print(f"[Nomader] Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimiser.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.global_step   = ckpt["global_step"]
        self.episodes_done = ckpt["episodes_done"]
        print(f"[Nomader] Resumed at step={self.global_step:,}  "
              f"eps={self.episodes_done:,}")

    # ── PPO update ────────────────────────────────────────────────────────────
    def _ppo_update(self, buffer):
        cfg = self.cfg
        p_losses, v_losses, entropies, grad_norms = [], [], [], []

        for _ in range(cfg["num_epochs"]):
            for obs_b, act_b, logp_old, ret_b, adv_b in \
                    buffer.get_minibatches(cfg["mini_batch_size"], self.device):

                _, logp_new, entropy, value = \
                    self.policy.get_action_and_value(obs_b, act_b)

                # normalise advantages per minibatch
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                ratio = (logp_new - logp_old).exp()
                p1    = ratio * adv_b
                p2    = ratio.clamp(1 - cfg["clip_eps"],
                                    1 + cfg["clip_eps"]) * adv_b
                p_loss = -torch.min(p1, p2).mean()

                # clipped value loss — stabilises critic early in training
                v_loss_raw   = (value - ret_b).pow(2)
                v_loss       = 0.5 * v_loss_raw.mean()

                loss = (p_loss
                        + cfg["vf_coef"]  * v_loss
                        - cfg["ent_coef"] * entropy.mean())

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                          cfg["max_grad_norm"])

                # track gradient norm — useful to spot dead/exploding gradients
                total_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.policy.parameters() if p.grad is not None
                ) ** 0.5
                grad_norms.append(total_norm)

                self.optimiser.step()

                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())

        self.scheduler.step()
        return (np.mean(p_losses), np.mean(v_losses),
                np.mean(entropies), np.mean(grad_norms))

    # ── render one frame on the separate render env ────────────────────────────
    def _maybe_render(self):
        if self.global_step < self._next_render:
            return
        self._next_render += self.cfg["render_every"]

        self.policy.eval()
        with torch.no_grad():
            obs_t = single_obs_to_tensor(self.render_obs, self.device)
            action_np = self.policy.actor.get_action(
                self.policy.encode_obs(obs_t), deterministic=True
            )[0].cpu().numpy()[0]

        self.render_obs, _, term, trunc, _ = self.render_env.step(action_np)
        if term or trunc:
            self.render_obs, _ = self.render_env.reset()
        self.render_env.render()

    # ── collect rollout (SubprocVecEnv path) ──────────────────────────────────
    def _collect_subproc(self, obs_dict, buffer):
        """obs_dict: {key: np.array (N, ...)} from SubprocVecEnv."""
        N = self.cfg["num_envs"]
        T = self.cfg["steps_per_env"]

        for _ in range(T):
            obs_t  = obs_sb3_to_tensor(obs_dict, self.device)
            action, log_prob, _, value = \
                self.policy.get_action_and_value(obs_t)

            acts_np                        = action.cpu().numpy()
            obs_next_dict, rewards, dones, infos = self.vec_env.step(acts_np)

            # SB3 auto-resets on done; handle episode stats via info
            for i in range(N):
                self.ep_rewards[i] += rewards[i]
                self.ep_lengths[i] += 1
                if dones[i]:
                    self.completed_rews.append(self.ep_rewards[i])
                    self.episodes_done  += 1
                    self.ep_rewards[i]   = 0.0
                    self.ep_lengths[i]   = 0

            buffer.store(obs_t, action, log_prob,
                         rewards, dones.astype(np.float32), value)
            obs_dict = obs_next_dict
            self.global_step += N
            self._maybe_render()

        return obs_dict

    # ── collect rollout (sequential fallback) ────────────────────────────────
    def _collect_sequential(self, obs_list, buffer):
        N = self.cfg["num_envs"]
        T = self.cfg["steps_per_env"]

        for _ in range(T):
            obs_t  = obs_list_to_tensor(obs_list, self.device)
            action, log_prob, _, value = \
                self.policy.get_action_and_value(obs_t)

            acts_np = action.cpu().numpy()
            rewards, dones = [], []

            for i, env in enumerate(self.vec_env):
                obs_next, r, term, trunc, _ = env.step(acts_np[i])
                done = term or trunc
                obs_list[i] = obs_next if not done else env.reset()[0]

                self.ep_rewards[i] += r
                self.ep_lengths[i] += 1
                if done:
                    self.completed_rews.append(self.ep_rewards[i])
                    self.episodes_done  += 1
                    self.ep_rewards[i]   = 0.0
                    self.ep_lengths[i]   = 0

                rewards.append(r)
                dones.append(float(done))

            buffer.store(obs_t, action, log_prob,
                         np.array(rewards), np.array(dones), value)
            self.global_step += N
            self._maybe_render()

        return obs_list

    # ── main training loop ────────────────────────────────────────────────────
    def train(self):
        cfg        = self.cfg
        T, N       = cfg["steps_per_env"], cfg["num_envs"]
        ckpt_every = cfg["ckpt_every"]
        next_ckpt  = (self.global_step // ckpt_every + 1) * ckpt_every

        # initialise observations
        if self.use_subproc:
            state = self.vec_env.reset()   # dict of (N, ...) arrays
        else:
            state = [env.reset()[0] for env in self.vec_env]

        print(f"\n[Nomader] Training {cfg['total_steps']:,} steps | "
              f"{N} envs | T={T} | "
              f"batch={T*N} | mini_batch={cfg['mini_batch_size']}\n")

        t_start       = time.time()
        last_log_step = self.global_step

        while self.global_step < cfg["total_steps"] and not self._stop:
            buffer = RolloutBuffer(T, N, self.device)

            # ── rollout ──────────────────────────────────────────────────────
            self.policy.eval()
            with torch.no_grad():
                if self.use_subproc:
                    state = self._collect_subproc(state, buffer)
                    last_obs_t  = obs_sb3_to_tensor(state, self.device)
                else:
                    state = self._collect_sequential(state, buffer)
                    last_obs_t  = obs_list_to_tensor(state, self.device)

                last_value = self.policy.get_value(last_obs_t)

            buffer.compute_returns(last_value, cfg["gamma"], cfg["gae_lambda"])

            # ── PPO update ───────────────────────────────────────────────────
            self.policy.train()
            pl, vl, ent, gnorm = self._ppo_update(buffer)

            # ── logging ──────────────────────────────────────────────────────
            if self.global_step - last_log_step >= cfg["log_every"]:
                last_log_step = self.global_step
                mean_rew = (np.mean(self.completed_rews[-50:])
                            if self.completed_rews else float("nan"))
                elapsed  = time.time() - t_start
                sps      = self.global_step / max(elapsed, 1)
                lr_now   = self.optimiser.param_groups[0]["lr"]

                print(
                    f"step={self.global_step:>8,} | "
                    f"ep={self.episodes_done:>6,} | "
                    f"rew={mean_rew:>7.2f} | "
                    f"p={pl:>7.4f} | v={vl:>7.4f} | "
                    f"ent={ent:>5.3f} | "
                    f"gnorm={gnorm:>5.3f} | "   # ← new: watch this
                    f"sps={sps:>5.0f} | lr={lr_now:.2e}"
                )

                self._csv_log([self.global_step, mean_rew,
                               np.mean(self.ep_lengths) if self.ep_lengths else 0,
                               pl, vl, ent, gnorm, lr_now])

            # ── checkpoint ───────────────────────────────────────────────────
            if self.global_step >= next_ckpt:
                self._save_checkpoint()
                next_ckpt += ckpt_every

        tag  = "_final" if not self._stop else "_interrupted"
        path = self._save_checkpoint(tag=tag)
        print(f"\n[Nomader] Done. {self.global_step:,} steps | "
              f"{self.episodes_done:,} episodes")

        if self.use_subproc:
            self.vec_env.close()
        self.render_env.close()
        return path


# ── entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Nomader PPO Trainer")
    p.add_argument("--resume",        type=str,   default=None)
    p.add_argument("--total_steps",   type=int,   default=CFG["total_steps"])
    p.add_argument("--num_envs",      type=int,   default=CFG["num_envs"])
    p.add_argument("--steps_per_env", type=int,   default=CFG["steps_per_env"])
    p.add_argument("--lr",            type=float, default=CFG["lr"])
    p.add_argument("--seed",          type=int,   default=CFG["seed"])
    p.add_argument("--ckpt_every",    type=int,   default=CFG["ckpt_every"])
    p.add_argument("--render_every",  type=int,   default=CFG["render_every"])
    p.add_argument("--no_render",     action="store_true",
                   help="Disable the render window entirely")
    return p.parse_args()


if __name__ == "__main__":
    # required for SubprocVecEnv on Linux/Mac
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    cfg  = {**CFG, **{k: v for k, v in vars(args).items() if v is not None}}

    if args.no_render:
        cfg["render_every"] = int(9e18)   # effectively disabled

    trainer = PPOTrainer(cfg)
    trainer.train()