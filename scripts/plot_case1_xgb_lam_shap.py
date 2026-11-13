from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from case1_xgb_lam import fit_selector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
TRAIN=ROOT.parent/'train.csv';FOLDS=ROOT.parent/'data'/'train_folds.csv';OUT=ROOT/'shap_figures'

def main():
    train=pd.read_csv(TRAIN);folds=pd.read_csv(FOLDS)
    assert len(train)==len(folds) and folds.fold.nunique()==10
    raw=train.drop(columns='SCA');raw=raw.drop(columns='nevents',errors='ignore');y=train.SCA.astype(int)
    S=pd.read_csv(OUT/'case1_xgb_lam_shap_oof.csv.gz')
    features=list(S.columns);V=np.full((len(train),len(features)),np.nan,dtype=np.float32)
    fold_ids=folds.fold.astype(int).to_numpy()
    for k in range(10):
        valid=np.flatnonzero(fold_ids==k);fit=np.flatnonzero(fold_ids!=k)
        imputer,columns,selected=fit_selector(raw.iloc[fit],y.iloc[fit])
        assert selected==features,'OOF SHAP columns differ from this fold feature selector.'
        V[valid]=imputer.transform(raw.iloc[valid][columns])[:,[columns.index(f) for f in features]]
    assert np.isfinite(V).all(),'Every row must have its out-of-fold feature values.'
    X=pd.DataFrame(V,columns=features)
    X.to_csv(OUT/'case1_xgb_lam_shap_values_X.csv.gz',index=False,compression='gzip')
    # Plot a seeded 30k-row sample for legibility; the saved SHAP/X matrices remain complete.
    rng=np.random.default_rng(42);idx=np.sort(rng.choice(len(X),min(30000,len(X)),replace=False))
    Ss=S.iloc[idx].to_numpy();Xs=X.iloc[idx]
    shap.summary_plot(Ss,Xs,max_display=11,show=False)
    ax=plt.gca();ax.axvline(0,color='#555555',linewidth=.8);ax.set_xlabel('SHAP value (log-odds contribution)')
    plt.title('Case 1 XGBoost - 10-fold OOF SHAP summary')
    plt.savefig(OUT/'case1_xgb_lam_shap_summary.png',dpi=200,bbox_inches='tight');plt.close()
    shap.summary_plot(Ss,Xs,plot_type='bar',max_display=11,show=False)
    ax=plt.gca();ax.set_xlabel('mean |SHAP| (log-odds contribution)')
    plt.title('Case 1 XGBoost - mean absolute OOF SHAP')
    plt.savefig(OUT/'case1_xgb_lam_shap_bar.png',dpi=200,bbox_inches='tight');plt.close()
    shap.dependence_plot('view_count',Ss,Xs,interaction_index='add_to_detail_ratio',show=False)
    ax=plt.gca();ax.axhline(0,color='#555555',linewidth=.8)
    ax.set_xlim(float(Xs.view_count.min()),float(Xs.view_count.quantile(.99)))
    ax.set_xlabel('view_count');ax.set_ylabel('SHAP value (log-odds contribution)')
    plt.title('SHAP dependence: view_count (color: add_to_detail_ratio)')
    plt.savefig(OUT/'case1_xgb_lam_shap_dependence_view_count.png',dpi=200,bbox_inches='tight');plt.close()
    print('SHAP:',shap.__version__,'rows:',len(S),'plot sample:',len(idx),'top feature:',S.abs().mean().sort_values(ascending=False).index[0])
if __name__=='__main__':main()
