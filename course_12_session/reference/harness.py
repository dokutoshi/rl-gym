"""Separate training seeds; raw artifacts first. No invented benchmark scores."""
import argparse, subprocess, sys, json, statistics
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--algo',default='q'); p.add_argument('--episodes',type=int,default=5000); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]); p.add_argument('--no-mask',action='store_true'); a=p.parse_args()
algos=['loop','mc','q','sarsa','linear','dqn','reinforce','ac','ppo','sft','dpo','grpo','evaluate'] if a.smoke else [a.algo]
summary=[]
for algo in algos:
    means=[]
    for seed in ([0] if a.smoke else a.seeds):
        out=Path('results')/('smoke' if a.smoke else 'benchmark')/algo/str(seed)
        command=[sys.executable,'-m','reference.train','--algo',algo,'--episodes',str(2 if a.smoke else a.episodes),'--seed',str(seed),'--out',str(out)]
        if a.no_mask: command+=['--no-mask']
        subprocess.run(command,check=True)
        means.append(json.loads((out/'metrics.json').read_text())['mean'])
    summary.append({'algo':algo,'run_means':means,'mean':statistics.mean(means),'sd_across_training_seeds':statistics.stdev(means) if len(means)>1 else None,'note':'Training scores, not held-out evaluation. Language scores differ in meaning by objective.'})
Path('results').mkdir(exist_ok=True); Path('results/summary.json').write_text(json.dumps(summary,indent=2))
