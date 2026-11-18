"""Render to a PNG by default, without a separate native display window."""
import argparse, struct, zlib
from pathlib import Path
import gymnasium as gym
p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['rgb_array','human'],default='rgb_array');p.add_argument('--steps',type=int,default=50);a=p.parse_args()
env=gym.make('CartPole-v1',render_mode=a.mode)
env.action_space.seed(0);env.reset(seed=0)
try:
    for _ in range(a.steps):
        _,_,term,trunc,_=env.step(env.action_space.sample())
        if term or trunc: env.reset()
    if a.mode=='rgb_array':
        pixels=env.render();h,w,_=pixels.shape
        def chunk(tag,data): return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
        raw=b''.join(b'\0'+row.tobytes() for row in pixels)
        png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
        Path('results').mkdir(exist_ok=True);Path('results/cartpole.png').write_bytes(png);print('Saved results/cartpole.png')
finally: env.close()
