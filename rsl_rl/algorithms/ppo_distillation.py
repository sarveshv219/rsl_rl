# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + Behaviour-Cloning (PPODistill) algorithm.

The student actor is trained with a combined objective::

    loss = surrogate_loss + value_loss_coef * value_loss
         + bc_coef(t) * bc_loss
         - entropy_coef * entropy

where ``bc_loss = MSE(actor_mean, teacher_mean)`` and ``bc_coef(t)`` decays
over training iterations according to a configurable schedule.

All modified/added sections are marked with ``# [PPODistill]``.
"""

from __future__ import annotations

import inspect
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.algorithms.ppo import PPO

# Resolved once at import time; filters isaaclab-injected extra keys from alg_cfg.
_PPO_INIT_PARAMS: frozenset[str] = frozenset(inspect.signature(PPO.__init__).parameters)

# ============================================================================ #
#  Scheduling helpers                                                           #
# ============================================================================ #

def _step_bc_coef(bc_coef: float, bc_coef_schedule: str, bc_coef_decay: float, bc_coef_min: float) -> float:
    if bc_coef_schedule == "linear":
        return max(bc_coef_min, bc_coef - bc_coef_decay)
    elif bc_coef_schedule == "exponential":
        return max(bc_coef_min, bc_coef * bc_coef_decay)
    elif bc_coef_schedule == "none":
        return bc_coef
    else:
        raise ValueError(f"Unknown bc_coef_schedule '{bc_coef_schedule}'")


# ============================================================================ #
#  PPODistillation                                                                    #
# ============================================================================ #

class PPODistillation(PPO):
    # [PPODistill] ---------------------------------------------------------------- #
    teacher: ActorCritic | ActorCriticRecurrent
    """Frozen teacher network."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        # [PPODistill] extra params mapped directly from on_policy_runner.py fallback -----------
        teacher_cfg: dict | None = None,
        obs: TensorDict | None = None,
        obs_groups: dict[str, list[str]] | None = None,
        num_actions: int | None = None,
        bc_coef: float = 1.0,
        bc_coef_schedule: str = "linear",
        bc_coef_decay: float = 1e-4,
        bc_coef_min: float = 0.0,
        teacher_loaded: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(policy, **{k: v for k, v in kwargs.items() if k in _PPO_INIT_PARAMS})

        if teacher_cfg is None:
            raise ValueError("PPODistillation requires a `teacher` configuration dict.")

        teacher_class = eval(teacher_cfg.pop("class_name", "ActorCritic"))
        # Build teacher-specific obs_groups: map both "policy" and "critic" to
        # the teacher's privileged obs group (kinematic-only), NOT the student's
        # visual "policy" group which has 1621-dim Theia+kinematic concatenated input.
        # This ensures the teacher ActorCritic gets the same 494-dim input it was
        # trained on.
        teacher_obs_key = obs_groups.get("teacher", obs_groups.get("critic", ["critic"]))
        teacher_obs_groups = {
            "policy": teacher_obs_key,
            "critic": teacher_obs_key,
        }
        self.teacher = teacher_class(obs, teacher_obs_groups, num_actions, **teacher_cfg).to(self.device)
        self.teacher.eval()  # Always frozen
        print(f"[PPODistill Teacher] Actor MLP: {self.teacher.actor}")
        print(f"[PPODistill Student] Actor MLP: {self.policy.actor}")

        self.bc_coef = float(bc_coef)
        self.bc_coef_schedule = bc_coef_schedule
        self.bc_coef_decay = float(bc_coef_decay)
        self.bc_coef_min = float(bc_coef_min)
        self.teacher_loaded = teacher_loaded

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and record frozen teacher means."""
        # Standard PPO act — fills transition.actions, values, log_probs, etc.
        student_actions = super().act(obs)

        # [PPODistill] Query teacher (frozen, no grad) ----------------------------- #
        with torch.no_grad():
            self.teacher.act(obs) # Ensure teacher processes input structurally
            teacher_mean = self.teacher.action_mean
        self.transition.teacher_actions = teacher_mean.detach()
        # --------------------------------------------------------------------- #

        return student_actions

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]) -> None:
        super().process_env_step(obs, rewards, dones, extras)
        self.teacher.reset(dones)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_bc_loss = 0.0  # [PPODistill]
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hidden_states_batch, \
            masks_batch, teacher_actions_batch = batch

            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Standard forward evaluation exactly like PPO
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # [PPODistill] Behaviour cloning loss ---------------------------------- #
            if teacher_actions_batch is not None and self.bc_coef > 0.0:
                teacher_targets = teacher_actions_batch[:original_batch_size].detach()
                bc_loss = nn.functional.mse_loss(mu_batch, teacher_targets)
            else:
                bc_loss = torch.tensor(0.0, device=self.device)
            # -----------------------------------------------------------------#

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.bc_coef * bc_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:])
                
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_bc_loss += bc_loss.item()  # [PPODistill]
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_bc_loss /= num_updates  # [PPODistill]
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        # [PPODistill] Advance schedule -------------------------------------------- #
        self.bc_coef = _step_bc_coef(self.bc_coef, self.bc_coef_schedule, self.bc_coef_decay, self.bc_coef_min)

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc": mean_bc_loss,
            "bc_coef": self.bc_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.teacher.eval() # Teacher must stay in eval

    def eval_mode(self) -> None:
        super().eval_mode()
        self.teacher.eval()

    def get_state_dict(self) -> dict:
        """v3.1.2 relies on get_state_dict rather than save()."""
        saved_dict = {}
        saved_dict["policy_state_dict"] = self.policy.state_dict()
        saved_dict["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()

        saved_dict["teacher_state_dict"] = self.teacher.state_dict()
        saved_dict["bc_coef"] = self.bc_coef
        return saved_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Handles PPO and RLBC states dynamically based on the 3.1.2 framework."""
        if "teacher_state_dict" in state_dict:
            self.policy.load_state_dict(state_dict["policy_state_dict"])
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            self.teacher.load_state_dict(state_dict["teacher_state_dict"])
            self.teacher_loaded = True
            if "bc_coef" in state_dict:
                self.bc_coef = float(state_dict["bc_coef"])
        elif "policy_state_dict" in state_dict:
            # Bootstrapping from classic checkpoint
            self.teacher.load_state_dict(state_dict["policy_state_dict"], strict=False)
            self.teacher_loaded = True

        if self.rnd and "rnd_state_dict" in state_dict:
            self.rnd.load_state_dict(state_dict["rnd_state_dict"])

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        model_params = [self.teacher.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.teacher.load_state_dict(model_params[0])
