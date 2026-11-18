"""n-step target: rewards start at R_(t+1); next_value is V(S_(t+n))."""
def nstep_target(rewards, n, gamma, next_value=0., terminal=False):
    if n < 1 or not rewards: raise ValueError('Need n>=1 and at least one reward')
    k=min(n,len(rewards))
    if k<n and not terminal: raise ValueError('Need n rewards or a terminal tail')
    result=sum(gamma**i*rewards[i] for i in range(k))
    return result if terminal and k==len(rewards) else result+gamma**k*next_value

if __name__=='__main__':
    for n in (1,2,3):
        print(n, nstep_target([1,2,3],n,.9,next_value=10,terminal=True))
