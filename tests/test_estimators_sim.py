"""Full comparison incl. PPI/difference estimator + bootstrap CI, under equal allocation."""
import numpy as np
from scipy import stats
from estimators import stratified_prop, hajek_prop, difference_prop, optimal_lam

sizes=np.array([136649,40994,13664,5465,2049,819,273,81])
pk=np.array([0.002,0.01,0.05,0.15,0.35,0.55,0.70,0.85])
strat=np.concatenate([np.full(s,k) for k,s in enumerate(sizes)])
r=np.random.default_rng(0)
y=(r.random(len(strat))<pk[strat]).astype(float)
# model prediction available EVERYWHERE (the AlphaEarth analogue)
yhat=np.clip(pk[strat]+r.normal(0,0.10,len(strat)),0,1)
TRUE=y.mean(); Nh={k:int(v) for k,v in enumerate(sizes)}; N=sizes.sum()
YHAT_POP=yhat.mean()   # known: model runs over the whole map

def strat_boot_ci(ys, ss, Nh, B=600, alpha=0.05, seed=0):
    """Stratified bootstrap: resample within strata, percentile CI."""
    rr=np.random.default_rng(seed); reps=[]
    idx_by={k:np.where(ss==k)[0] for k in np.unique(ss)}
    for b in range(B):
        bidx=np.concatenate([rr.choice(v,size=len(v),replace=True) for v in idx_by.values()])
        reps.append(stratified_prop(ys[bidx], ss[bidx], Nh)[0])
    return np.percentile(reps,[100*alpha/2,100*(1-alpha/2)])

R=400
res={k:{'est':[],'cov':0,'w':[]} for k in ['stratified','strat_boot','ppi_lam1','ppi_opt','hajek']}
for i in range(R):
    rr=np.random.default_rng(5000+i); idx=[]
    for k in Nh:
        pool=np.where(strat==k)[0]; idx.append(rr.choice(pool,size=min(100,len(pool)),replace=False))
    idx=np.concatenate(idx)
    ys,ss,yh=y[idx],strat[idx],yhat[idx]
    nh={k:(ss==k).sum() for k in Nh}
    w=np.array([Nh[s]/nh[s] for s in ss])

    p,se,ci=stratified_prop(ys,ss,Nh); res['stratified']['est'].append(p)
    res['stratified']['cov']+= (ci[0]<=TRUE<=ci[1]); res['stratified']['w'].append(ci[1]-ci[0])

    bci=strat_boot_ci(ys,ss,Nh,B=400,seed=i)
    res['strat_boot']['est'].append(p); res['strat_boot']['cov']+=(bci[0]<=TRUE<=bci[1]); res['strat_boot']['w'].append(bci[1]-bci[0])

    p3,se3,ci3=difference_prop(ys,yh,YHAT_POP,w=w,lam=1.0); res['ppi_lam1']['est'].append(p3)
    res['ppi_lam1']['cov']+=(ci3[0]<=TRUE<=ci3[1]); res['ppi_lam1']['w'].append(ci3[1]-ci3[0])

    lam=optimal_lam(ys,yh,w=w)
    p4,se4,ci4=difference_prop(ys,yh,YHAT_POP,w=w,lam=lam); res['ppi_opt']['est'].append(p4)
    res['ppi_opt']['cov']+=(ci4[0]<=TRUE<=ci4[1]); res['ppi_opt']['w'].append(ci4[1]-ci4[0])

    p5,se5,ci5=hajek_prop(ys,w); res['hajek']['est'].append(p5)
    res['hajek']['cov']+=(ci5[0]<=TRUE<=ci5[1]); res['hajek']['w'].append(ci5[1]-ci5[0])

print(f'TRUE = {TRUE:.5f}   (rare-event regime, equal allocation, {R} reps)\n')
print(f"{'estimator':<14}{'mean':>10}{'bias':>11}{'coverage':>11}{'mean CI width':>15}")
print('-'*61)
for k,v in res.items():
    m=np.mean(v['est'])
    print(f"{k:<14}{m:>10.5f}{m-TRUE:>+11.5f}{v['cov']/R:>10.1%}{np.mean(v['w']):>15.5f}")
print()
print('lam* (PPI++ power tuning) mean =', np.mean([optimal_lam(y[np.concatenate([np.random.default_rng(5000+i).choice(np.where(strat==k)[0],size=min(100,Nh[k]),replace=False) for k in Nh])], yhat[np.concatenate([np.random.default_rng(5000+i).choice(np.where(strat==k)[0],size=min(100,Nh[k]),replace=False) for k in Nh])]) for i in range(20)]))
