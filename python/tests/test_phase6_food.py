from __future__ import annotations

from flybrain.env_food_search import FakeFoodSearchEnv
from flybrain.phase6 import (
    AGENT_OBS_HAS_FOOD_XYZ,
    EXPECTED_AGENT_VECTOR_SIZE,
    HUNGER_SEARCH_THRESHOLD,
    MIN_SUGAR_DISTANCE_REDUCTION,
    SATED_ENERGY_THRESHOLD,
)
from flybrain.training.rewards import HomeostaticRewardConfig, compute_total_reward
from flybrain.unity_env import DEFAULT_OBS_DIM


def test_phase6_constants() -> None:
    assert AGENT_OBS_HAS_FOOD_XYZ is False
    assert EXPECTED_AGENT_VECTOR_SIZE == DEFAULT_OBS_DIM == 197


def test_homeostatic_reward_sugar_and_bitter() -> None:
    base = compute_total_reward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sweet = compute_total_reward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sugar_consume=0.1)
    bitter = compute_total_reward(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, bitter_contact=1.0)
    cfg = HomeostaticRewardConfig(sugar_consume_weight=2.0, bitter_penalty_weight=1.5)
    assert sweet > base
    assert bitter < base
    assert cfg.sugar_consume_weight == 2.0


def test_fake_food_search_no_food_xyz_in_obs() -> None:
    env = FakeFoodSearchEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (DEFAULT_OBS_DIM,)
    assert info["has_food_xyz_in_obs"] is False
    assert "_privileged_sugar" in info
    # Ensure target coords are not appended to obs
    assert obs.shape[0] == EXPECTED_AGENT_VECTOR_SIZE


def test_hungry_agent_approaches_and_consumes_sugar() -> None:
    env = FakeFoodSearchEnv()
    obs, info = env.reset(seed=11)
    start = info["distance_to_sugar"]
    energy0 = info["energy"]
    assert info["hunger"] >= HUNGER_SEARCH_THRESHOLD - 0.2
    for _ in range(100):
        action = env.food_policy_action(hungry=True)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert info["has_food_xyz_in_obs"] is False
    improved = info["distance_to_sugar"] <= start - MIN_SUGAR_DISTANCE_REDUCTION
    fed = info["sugar_consumed"] > 0.0 or info["energy"] > energy0
    assert improved or fed


def test_distinguish_sugar_vs_bitter_actions() -> None:
    env = FakeFoodSearchEnv()
    env.reset(seed=2)
    # Place agent on sugar vs bitter by teleporting privileged state
    env._pos = env._sugar.copy()
    env._refresh_distances()
    on_sugar = env.food_policy_action(hungry=True)
    env._pos = env._bitter.copy()
    env._refresh_distances()
    on_bitter = env.food_policy_action(hungry=True)
    assert on_sugar[4] > on_bitter[4]  # more proboscis on sugar
    assert abs(on_bitter[1]) >= 0.2 or on_bitter[0] < on_sugar[0]


def test_sated_vs_hungry_behavior_differs() -> None:
    env = FakeFoodSearchEnv()
    env.reset(seed=5)
    env._energy = 0.2
    env._hunger = 0.8
    hungry_act = env.food_policy_action(hungry=True)
    env._energy = 0.9
    env._hunger = 0.1
    sated_act = env.food_policy_action(hungry=False)
    assert hungry_act[0] > sated_act[0]
    assert env._energy >= SATED_ENERGY_THRESHOLD
