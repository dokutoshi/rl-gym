"""CPU teaching implementations. Run --help. No downloads or pretrained models."""
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


def neural(args):
    import torch
    from torch import nn
    from torch.distributions import Categorical
    torch.manual_seed(args.seed); torch.set_num_threads(1)
    rng=np.random.default_rng(args.seed); random.seed(args.seed)
    env=gym.make(args.env or 'CartPole-v1')
    dim=env.observation_space.shape[0]; n=env.action_space.n
    linear=args.algo=='linear'
    model=nn.Linear(dim,n) if linear else nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,n))
    critic=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
    optimizer=torch.optim.Adam(list(model.parameters())+list(critic.parameters()),lr=args.lr)
    if args.resume:
        ck=torch.load(args.resume,weights_only=False); model.load_state_dict(ck['model']); critic.load_state_dict(ck['critic']); optimizer.load_state_dict(ck['optimizer'])
    frozen=copy.deepcopy(model); replay=deque(maxlen=10000); scores=[]; updates=0
    tensor=lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32)
    for ep in range(args.episodes):
        s,_=env.reset(seed=args.seed+ep); rollout=[]; total=0
        for step in range(args.max_steps):
            x=tensor(s); logits=model(x)
            if args.algo in ('linear','dqn'):
                a=int(rng.integers(n)) if rng.random()<args.epsilon else int(logits.argmax())
            else: a=int(Categorical(logits=logits).sample())
            ns,r,term,trunc,_=env.step(a); total+=r
            rollout.append((s,a,r,ns,term,trunc))
            if args.algo in ('linear','dqn'):
                replay.append((s,a,r,ns,term))
                batch=random.sample(list(replay),min(32,len(replay))) if not linear else [replay[-1]]
                bs,ba,br,bn,bt=zip(*batch)
                pred=model(tensor(bs)).gather(1,torch.tensor(ba)[:,None]).squeeze(1)
                with torch.no_grad():
                    nxt=(model if linear else frozen)(tensor(bn)).max(1).values
                    y=tensor(br)+args.gamma*(1-tensor(bt))*nxt
                loss=nn.functional.smooth_l1_loss(pred,y)
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
                updates+=1
                if updates%100==0: frozen.load_state_dict(model.state_dict())
            s=ns
            if term or trunc: break
        if args.algo not in ('linear','dqn'):
            states=tensor([t[0] for t in rollout]); actions=torch.tensor([t[1] for t in rollout])
            with torch.no_grad():
                # Bootstrap truncations; never bootstrap true terminations.
                g=0. if (rollout[-1][4] or args.algo=='reinforce') else float(critic(tensor(rollout[-1][3])).item())
                ret=[]
                for t in reversed(rollout): g=t[2]+args.gamma*g; ret.append(g)
                returns=tensor(ret[::-1]); values=critic(states).squeeze(1)
                adv=returns-values if args.algo in ('ppo','ac') else returns
                old=Categorical(logits=model(states)).log_prob(actions)
            for _ in range(4 if args.algo=='ppo' else 1):
                dist=Categorical(logits=model(states)); lp=dist.log_prob(actions)
                if args.algo=='ppo':
                    ratio=(lp-old).exp(); policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
                else: policy=-(lp*adv).mean()
                value_loss=(critic(states).squeeze(1)-returns).square().mean() if args.algo in ('ppo','ac') else tensor(0.)
                loss=policy+.5*value_loss-.01*dist.entropy().mean()
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(model.parameters())+list(critic.parameters()),.5); optimizer.step()
        scores.append(total)
        if (ep+1)%args.save_every==0 or ep==args.episodes-1:
            Path(args.out).mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'critic':critic.state_dict(),'optimizer':optimizer.state_dict(),'episode':ep+1,'config':vars(args)},Path(args.out,'checkpoint.pt'))
    env.close(); return scores


def language(args):
    """One-token conditional language model: exact objectives, tiny vocabulary.
    Not a transformer or a benchmark of general language ability.
    """
    import torch
    torch.manual_seed(args.seed); torch.set_num_threads(1)
    # Four prompt IDs, three completion tokens. Correct token is prompt % 3.
    logits=torch.nn.Parameter(torch.zeros(4,3)); optimizer=torch.optim.Adam([logits],lr=args.lr)
    if args.resume:
        ck=torch.load(args.resume,weights_only=False); logits.data.copy_(ck['logits']); optimizer.load_state_dict(ck['optimizer'])
    ref=ck['reference_logprobs'].clone() if args.resume else logits.detach().clone().log_softmax(-1); history=[]
    for step in range(args.episodes):
        lp=logits.log_softmax(-1); ids=torch.arange(4); chosen=ids%3; rejected=(chosen+1)%3
        if args.algo=='sft': loss=-lp[ids,chosen].mean()
        elif args.algo=='dpo':
            margin=(lp[ids,chosen]-lp[ids,rejected])-(ref[ids,chosen]-ref[ids,rejected])
            loss=-torch.nn.functional.logsigmoid(args.beta*margin).mean()
        else:
            with torch.no_grad():
                old=logits.detach().log_softmax(-1)
                actions=torch.distributions.Categorical(logits=logits).sample((8,)).T
                rewards=(actions==chosen[:,None]).float()
                adv=(rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True,unbiased=False)+1e-8)
                old_a=old.gather(1,actions); ref_a=ref.gather(1,actions)
            # Group-relative clipped surrogate, sampled positive KL estimator.
            for _ in range(4):
                new=logits.log_softmax(-1).gather(1,actions); ratio=(new-old_a).exp()
                kl=(ref_a-new).exp()-(ref_a-new)-1
                loss=-(torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)-args.beta*kl).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            history.append(float(rewards.mean())); continue
        optimizer.zero_grad(); loss.backward(); optimizer.step(); history.append(float(-loss.detach()))
    Path(args.out).mkdir(parents=True,exist_ok=True)
    torch.save({'logits':logits.detach(),'optimizer':optimizer.state_dict(),'reference_logprobs':ref,'config':vars(args)},Path(args.out,'checkpoint.pt'))
    return history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algo',choices=['loop','mc','q','sarsa','evaluate','linear','dqn','reinforce','ac','ppo','sft','dpo','grpo'],required=True)
    p.add_argument('--env'); p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=.01); p.add_argument('--gamma',type=float,default=.99); p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--beta',type=float,default=.1); p.add_argument('--no-mask',action='store_true'); p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--out',default='results/run'); p.add_argument('--resume'); p.add_argument('--save-every',type=int,default=10)
    args=p.parse_args(); start=time.time()
    if args.episodes<1: p.error('episodes must be positive')
    if args.algo in ('loop','mc','q','sarsa','evaluate'): scores=tabular(args)
    elif args.algo in ('sft','dpo','grpo'): scores=language(args)
    else: scores=neural(args)
    import platform, importlib.metadata
    result={'config':vars(args),'scores':scores,'mean':float(np.mean(scores)),'seconds':time.time()-start,'python':platform.python_version(),'versions':{k:importlib.metadata.version(k) for k in ['numpy','gymnasium','torch']}}
    Path(args.out).mkdir(parents=True,exist_ok=True); Path(args.out,'metrics.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='scores'}))

if __name__=='__main__': main()
