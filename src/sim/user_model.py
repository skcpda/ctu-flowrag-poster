"""Synthetic user model for automated bandit evaluation.

This stub lets us train / test the `BanditAgent` (currently uniform random)
without human feedback by sampling rewards according to pre-defined preferences.

Usage (example):

```python
from src.bandit.bandit_agent import BanditAgent
from src.sim.user_model import SyntheticUserModel, run_bandit_simulation

arms = [str(i) for i in range(10)]
agent = BanditAgent(arms)
user = SyntheticUserModel(preferences={"0": 0.9, "3": 0.8})
run_bandit_simulation(agent, user, context="eligibility", rounds=1000)
```
"""

from __future__ import annotations

import random
from typing import Dict, List


class SyntheticUserModel:
    """A simple probabilistic click / reward generator.

    Each arm has an intrinsic click-through probability *p* in `[0,1]`.
    During `sample_reward` we draw from Bernoulli(p) and return the result
    as the reward (1 = click / positive, 0 = negative).
    """

    def __init__(self, preferences: Dict[str, float], noise: float = 0.05):
        """Create a user model.

        Args:
            preferences: Mapping `arm → base_probability`.
            noise: Std-dev of uniform noise added per sample to avoid
                   deterministic behaviour (default 0.05).
        """
        self.preferences = preferences
        self.noise = noise

    def click_probability(self, arm: str) -> float:
        """Return current click probability for the given arm."""
        base = self.preferences.get(arm, 0.1)  # unseen arms → low prob
        # inject small noise so probability ∈ (0,1)
        return max(0.0, min(1.0, base + random.uniform(-self.noise, self.noise)))

    def sample_reward(self, arm: str) -> float:
        """Sample a Bernoulli reward (0/1) for the chosen arm."""
        p = self.click_probability(arm)
        return 1.0 if random.random() < p else 0.0


def run_bandit_simulation(
    agent,  # BanditAgent or any object with choose_arm & record(update)
    user: SyntheticUserModel,
    context: str,
    rounds: int = 1000,
    verbose: bool = False,
):
    """Run a closed-loop simulation: agent selects arms, user returns reward.

    Args:
        agent: The BanditAgent instance we want to evaluate.
        user: SyntheticUserModel that generates rewards.
        context: Context string passed to the bandit for each round.
        rounds: Number of interaction rounds.
        verbose: Print progress if True.
    """
    for t in range(1, rounds + 1):
        arm = agent.choose_arm(context)
        reward = user.sample_reward(arm)
        agent.record(context, arm, reward)
        if verbose and t % 100 == 0:
            print(f"Round {t}: arm={arm} reward={reward:.0f}") 