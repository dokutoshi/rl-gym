"""Sessions 2–4: tables, Monte Carlo and TD. No neural networks."""
import argparse, copy, json, random, time
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import gymnasium as gym


def target(reward, terminated, next_value, gamma=.99):
    return reward + gamma * (not terminated) * next_value


def choose(q, mask, epsilon, rng):
    valid = np.flatnonzero(mask)
    if not len(valid):
        raise ValueError('No valid actions in nonterminal state')
    if rng.random() < epsilon:
        return int(rng.choice(valid))
    best = valid[q[valid] == q[valid].max()]
    return int(rng.choice(best))


def tabular(args):
    env = gym.make(args.env or {'loop':'CartPole-v1','mc':'Blackjack-v1', 'q':'Taxi-v4','sarsa':'CliffWalking-v1','evaluate':'FrozenLake-v1'}[args.algo])
    rng = np.random.default_rng(args.seed)
    n = env.action_space.n
    q = defaultdict(lambda: np.zeros(n)); counts = defaultdict(lambda: np.zeros(n))
    if args.resume:
        import ast
        saved = json.loads(Path(args.resume).read_text())
        for key, value in saved['q'].items(): q[ast.literal_eval(key)] = np.array(value)
        for key, value in saved['counts'].items(): counts[ast.literal_eval(key)] = np.array(value)
    scores=[]
    for ep in range(args.episodes):
        s, info = env.reset(seed=args.seed+ep)
        key = lambda x: tuple(float(v) for v in x) if isinstance(x, (tuple, np.ndarray)) else int(x)
        def act(state, inf):
            mask = inf.get('action_mask', np.ones(n)) if not args.no_mask else np.ones(n)
            return choose(q[key(state)], mask, 0 if args.algo=='evaluate' else args.epsilon, rng)
        a = act(s,info) if args.algo!='loop' else int(rng.integers(n))
        trajectory=[]; total=0
        for step in range(args.max_steps):
            ns,r,term,trunc,ni = env.step(a); total+=r
            trajectory.append((key(s),a,r))
            na = act(ns,ni) if not term and args.algo!='loop' else 0
            if args.algo in ('q','sarsa'):
                mask = ni.get('action_mask',np.ones(n)) if not args.no_mask else np.ones(n)
                valid=np.flatnonzero(mask)
                nv=0 if term else (q[key(ns)][na] if args.algo=='sarsa' else q[key(ns)][valid].max())
                q[key(s)][a] += args.lr*(target(r,term,nv,args.gamma)-q[key(s)][a])
            s,info,a=ns,ni,na if args.algo!='loop' else int(rng.integers(n))
            if term or trunc: break
        if args.algo=='mc':
            # First-visit MC control; reverse returns, forward first-visit filter.
            returns=[]; g=0
            for _,_,r in reversed(trajectory): g=r+args.gamma*g; returns.append(g)
            seen=set()
            for (st,ac,_),g in zip(trajectory,reversed(returns)):
                if (st,ac) in seen: continue
                seen.add((st,ac)); counts[st][ac]+=1
                q[st][ac]+=(g-q[st][ac])/counts[st][ac]
        scores.append(total)
    env.close()
    Path(args.out).mkdir(parents=True,exist_ok=True)
    Path(args.out,'table.json').write_text(json.dumps({'q':{repr(k):v.tolist() for k,v in q.items()},'counts':{repr(k):v.tolist() for k,v in counts.items()}}))
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algo',choices=['mc','q','sarsa','evaluate'],required=True)
    p.add_argument('--env');p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--gamma',type=float,default=.99);p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--no-mask',action='store_true');p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--resume');p.add_argument('--out',default='results/tabular')
    args=p.parse_args()
    if args.episodes<1:p.error('episodes must be positive')
    scores=tabular(args)
    Path(args.out,'metrics.json').write_text(json.dumps({'config':vars(args),'scores':scores,'mean':float(np.mean(scores))},indent=2))
    print(f'Collected {len(scores)} episodes; mean return {np.mean(scores):.2f}')

if __name__=='__main__':main()
