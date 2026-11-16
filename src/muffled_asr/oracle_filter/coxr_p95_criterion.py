import pandas as pd, numpy as np
t=pd.read_csv('results/p0/tables/coxr_pairs.csv', keep_default_na=False, na_values=[''])
rng=np.random.default_rng(0); B=2000
def boot_pooled(s):
    out=[]; fams=[g for _,g in s.groupby('mask')]
    for _ in range(B):
        tot=0; W=0
        for g in fams:
            cu=np.bincount(rng.integers(1,7,6),minlength=7); cm=np.bincount(rng.integers(1,7,6),minlength=7)
            w=cu[g.u_read.values]*cm[g.m_read.values]; tot+=(g.gap.values*w).sum(); W+=w.sum()
        if W: out.append(tot/W)
    return np.array(out)
rows=[]
for snr in (0,-5,-10,99):
    for mask in ['cloth_v1_acetate','printed_cloth','surgical','all']:
        s=t[(t.snr==snr)&((t['mask']==mask) if mask!='all' else True)]
        nu,ma=s[s.kind=='null'],s[s.kind=='masked']; bn=boot_pooled(nu); bm=boot_pooled(ma)
        rows.append(dict(snr='quiet' if snr==99 else snr,mask=mask,n_null=len(nu),n_masked=len(ma),
          WER_U_masked=round(ma.U.mean(),2), masked_gap=round(ma.gap.mean(),2), masked_gap_CI=list(np.percentile(bm,[2.5,97.5]).round(2)),
          null_pair_median=round(nu.gap.median(),2), null_pair_p95=round(np.percentile(nu.gap,95),2),
          null_pooled_median=round(np.median(bn),2), null_pooled_p95=round(np.percentile(bn,95),2),
          masked_resid=round(ma.residual.mean(),2), null_resid=round(nu.residual.mean(),2),
          exceeds_pair_p95=bool(ma.gap.mean()>np.percentile(nu.gap,95)),
          exceeds_pooled_p95=bool(ma.gap.mean()>np.percentile(bn,95))))
T=pd.DataFrame(rows); pd.set_option('display.width',260)
T.to_csv('results/p0/tables/coxr_floor_p95_criterion.csv',index=False); print(T.to_string(index=False))
