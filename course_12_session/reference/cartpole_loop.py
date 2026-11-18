"""Session 1: the Gymnasium loop. Random actions, no learning or neural network."""
import argparse
import json
from pathlib import Path
import gymnasium as gym

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render-mode', choices=['human','rgb_array'], default=None)
    parser.add_argument('--out', default='results/01_loop')
    args=parser.parse_args()
    env=gym.make('CartPole-v1', render_mode=args.render_mode)
    env.action_space.seed(args.seed)
    scores=[]; endings=[]
    try:
        for episode in range(args.episodes):
            observation, info=env.reset(seed=args.seed+episode)
            total_reward=0
            while True:
                action=env.action_space.sample()  # Random choice; nothing is learned.
                observation, reward, terminated, truncated, info=env.step(action)
                total_reward+=reward
                if terminated or truncated:
                    endings.append({'terminated':bool(terminated),'truncated':bool(truncated)})
                    break
            scores.append(total_reward)
    finally:
        env.close()
    output=Path(args.out); output.mkdir(parents=True,exist_ok=True)
    output.joinpath('metrics.json').write_text(json.dumps({'config':vars(args),'scores':scores,'endings':endings},indent=2))
    print(f'Collected {len(scores)} episodes. Random-policy mean return: {sum(scores)/len(scores):.1f}')

if __name__=='__main__': main()
