"""Session 12: Monte Carlo policy evaluation • capstone harness. Runnable scaffold; implement student_update then add it to train.py."""
import subprocess, sys

def student_update(reward, terminated, next_value, gamma=.99):
    # TODO: replace this baseline target; compare against reference target().
    return reward

if __name__ == '__main__':
    subprocess.run([sys.executable,'-m','reference.train','--algo','evaluate','--episodes','2','--out','results/starter_12'],check=True)
