import time
from balatro_bot.dqn_factored.train import train
from balatro_bot.dqn_factored.eval import evaluate
from jackdaw.env.rewards import ShapedReward

t0 = time.time()
agent = train(total_steps=100_000, buffer_capacity=100_000, seed=0, reward_fn=ShapedReward(loss=0.0))
print("TRAIN_ELAPSED", time.time() - t0)

t0 = time.time()
res = evaluate(agent, n_episodes=30)
print("EVAL_ELAPSED", time.time() - t0)
print("EVAL_RESULTS", res)
