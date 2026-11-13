from jackdaw.env import BalatroEnvironment, DirectAdapter

NUM_EPISODES = 1

env = BalatroEnvironment(adapter_factory=DirectAdapter)
# from greedy_engine import GreedyAgent
from greedy.greedy_engine import GreedyAgent
agent = GreedyAgent()
results = []
print("=== Greedy Agent ===")
for ep in range(NUM_EPISODES):
    obs, mask, info = env.reset()
    agent.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(
            obs={"global": obs.global_context, **obs.entities},
            action_mask=mask,
            info=info,
        )
        obs, terminated, truncated, mask, info = env.step(action)
    results.append((env.episode_ante, env.episode_won, env.episode_length))
    print(f"  Episode {ep}: ante={env.episode_ante}, won={env.episode_won}, steps={env.episode_length}")

for ante, won, steps in results:
    print(f"  Episode: ante={ante}, won={won}, steps={steps}")


