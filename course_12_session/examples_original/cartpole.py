import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"step={step:4d} | action={action} | reward={reward:.1f} | obs={observation}", flush=True)

    if terminated or truncated:
        reason = "FELL" if terminated else "survived-500"
        print(f"--- reset at step {step} | {reason} ---", flush=True)
        observation, info = env.reset()

env.close()