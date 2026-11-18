"""Maskable PPO policy that scores each semantic action candidate."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableDistribution


class GameStateFeaturesExtractor(BaseFeaturesExtractor):
    """Encode game state while leaving candidate actions for the actor."""

    def __init__(self, observation_space: spaces.Dict):
        state_spaces = {
            key: value
            for key, value in observation_space.spaces.items()
            if key != "candidate_action"
        }
        if not state_spaces:
            raise ValueError("candidate_action must be accompanied by game-state features")
        state_extractor = CombinedExtractor(spaces.Dict(state_spaces))
        super().__init__(observation_space, state_extractor.features_dim)
        self.state_extractor = state_extractor

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        state_observations = {
            key: value for key, value in observations.items() if key != "candidate_action"
        }
        return self.state_extractor(state_observations)


class CandidateActionActorCriticPolicy(MaskableActorCriticPolicy):
    """Use a shared scorer over (policy context, action candidate) pairs.

    Unlike a static 500-logit output layer, the same scorer handles every
    action row, so an action slot's semantics can change between states.
    """

    def __init__(self, *args: Any, candidate_hidden_dim: int = 64, **kwargs: Any):
        self.candidate_hidden_dim = int(candidate_hidden_dim)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        candidate_dim = int(self.observation_space.spaces["candidate_action"].shape[-1])
        hidden = self.candidate_hidden_dim
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, hidden), nn.Tanh(),
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            for module in (
                self.features_extractor,
                self.mlp_extractor,
                self.candidate_encoder,
                self.context_encoder,
            ):
                module.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.action_scorer.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.action_scorer[-1].apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1.0))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _candidate_distribution(
        self, observations: dict[str, th.Tensor], latent_pi: th.Tensor
    ) -> MaskableDistribution:
        candidates = observations["candidate_action"].float()
        candidate_embeddings = self.candidate_encoder(candidates)
        context = self.context_encoder(latent_pi).unsqueeze(1)
        context = context.expand(-1, candidate_embeddings.shape[1], -1)
        logits = self.action_scorer(
            th.cat((context, candidate_embeddings), dim=-1)
        ).squeeze(-1)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(
        self,
        obs: dict[str, th.Tensor],
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._candidate_distribution(obs, latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: dict[str, th.Tensor],
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._candidate_distribution(obs, latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(
        self, obs: dict[str, th.Tensor], action_masks: np.ndarray | None = None
    ) -> MaskableDistribution:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._candidate_distribution(obs, latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def _predict(
        self,
        observation: dict[str, th.Tensor],
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> th.Tensor:
        return self.get_distribution(observation, action_masks).get_actions(
            deterministic=deterministic
        )
