from jackdaw.env import BalatroEnvironment, DirectAdapter, RandomAgent

env = BalatroEnvironment(adapter_factory=DirectAdapter)
agent = RandomAgent()

for episode in range(10):
    obs, mask, info = env.reset()
    agent.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(
            obs={"global": obs.global_context, **obs.entities},
            action_mask=mask,
            info=info,
        )
        obs, terminated, truncated, mask, info = env.step(action)

    print(f"Episode {episode}: ante={env.episode_ante}, won={env.episode_won}, steps={env.episode_length}")
