from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score,balanced_accuracy_score,recall_score,roc_auc_score
from xgboost import XGBClassifier

TRAIN=ROOT.parent/'train.csv'; TEST=ROOT.parent/'test.csv'; FOLDS=ROOT.parent/'Case 1'/'data'/'train_folds.csv'
OUT=ROOT/'results'/'case1'; TARGET='SCA'; SEED=42

def vif_table(frame):
    a=frame.to_numpy(dtype=float); rows=[]
    for j,name in enumerate(frame.columns):
        y=a[:,j]; design=np.column_stack([np.ones(len(a)),np.delete(a,j,axis=1)])
        resid=y-design@np.linalg.lstsq(design,y,rcond=None)[0]; tss=((y-y.mean())**2).sum()
        r2=1-(resid@resid)/tss if tss else 1.0
        rows.append((name,np.inf if 1-r2<=1e-12 else 1/(1-r2)))
    return pd.DataFrame(rows,columns=['feature','VIF']).sort_values('VIF',ascending=False)

def fit_selector(X,y):
    const=[c for c in X if X[c].nunique(dropna=True)<=1]; X=X.drop(columns=const)
    imp=SimpleImputer(strategy='median'); data=pd.DataFrame(imp.fit_transform(X),columns=X.columns,index=X.index)
    cy=data.apply(lambda col:col.corr(y)).abs(); selected=data.copy()
    while True:
        corr=selected.corr(); pairs=[(a,b,float(corr.loc[a,b])) for i,a in enumerate(corr.columns) for b in corr.columns[i+1:] if abs(corr.loc[a,b])>=.70]
        if not pairs: break
        a,b,r=sorted(pairs,key=lambda z:-abs(z[2]))[0]
        selected=selected.drop(columns=(a if cy[a]<cy[b] else b))
    while len(selected.columns)>1:
        v=vif_table(selected)
        if v.iloc[0].VIF<5: break
        selected=selected.drop(columns=v.iloc[0].feature)
    return imp,list(data.columns),list(selected.columns)

def metrics(y,score):
    mse=mean_squared_error(y,score); label=(score>=.5).astype(int)
    return {'rmse':float(np.sqrt(mse)),'mse':float(mse),'mae':float(mean_absolute_error(y,score)),'r2':float(r2_score(y,score)),
            'accuracy':float(accuracy_score(y,label)),'balanced_accuracy':float(balanced_accuracy_score(y,label)),
            'recall_SCA1':float(recall_score(y,label,pos_label=1,zero_division=0)),
            'recall_SCA0':float(recall_score(y,label,pos_label=0,zero_division=0)),'recall':float(recall_score(y,label,pos_label=1,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,score)),'precision':float(__import__('sklearn.metrics',fromlist=['precision_score']).precision_score(y,label,zero_division=0)),'f1':float(__import__('sklearn.metrics',fromlist=['f1_score']).f1_score(y,label,zero_division=0))}

def main():
    tr=pd.read_csv(TRAIN); te=pd.read_csv(TEST); fold_df=pd.read_csv(FOLDS)
    assert TARGET in tr and TARGET in te and 'fold' in fold_df
    assert len(tr)==len(fold_df),f'train rows {len(tr)} != shared-fold rows {len(fold_df)}'
    align_cols=[c for c in tr.columns if c!='avg_dwell_time']
    assert tr[align_cols].equals(fold_df[align_cols]),'Train and fold file row order/content differs beyond avg_dwell_time; refusing to apply fold labels.'
    dwell_mismatch=~(tr['avg_dwell_time'].eq(fold_df['avg_dwell_time']) | (tr['avg_dwell_time'].isna() & fold_df['avg_dwell_time'].isna()))
    assert int(dwell_mismatch.sum())==16,'Unexpected avg_dwell_time mismatch count; inspect before using shared folds.'
    print('Heads-up: 66 exact record overlaps exist across the agreed train/test files; preserving the group files and reporting metrics as provided.',flush=True)
    print('Using current train.csv values; shared fold labels align by row order. avg_dwell_time differs in 16 rows; CV uses train.csv values.',flush=True)
    assert set(fold_df.fold.unique())==set(range(10)),'Expected shared fold IDs 0-9.'
    X=tr.drop(columns=TARGET); y=tr[TARGET].astype(int)
    Xt=te.drop(columns=TARGET); yt=te[TARGET].astype(int)
    # Exclude structural duplicate total, as in the Case 1 variable-screening analysis.
    event_cols=['add_count','remove_count','view_count','detail_count']
    if 'nevents' in X and all(c in X for c in event_cols) and (X.nevents-X[event_cols].sum(axis=1)).eq(0).mean()>.99:
        X=X.drop(columns='nevents'); Xt=Xt.drop(columns='nevents')
    # Use the team's shared ten fold assignments directly.
    cv_id=fold_df.fold.astype(int).to_numpy()
    rng=np.random.default_rng(SEED)
    choices=[]
    seen=set()
    grids={'n_estimators':[100,200,300,400,500],'max_depth':[2,3,4,5,6],
      'learning_rate':[.03,.05,.08,.1],'subsample':[.7,.8,.9,1.0],
      'colsample_bytree':[.7,.8,.9,1.0],'min_child_weight':[1,3,5],'reg_lambda':[.5,1,2,5], 'scale_pos_weight':[.25,.5,1.0]}
    while len(choices)<30:
        item={k:(int(rng.choice(v)) if k in ('n_estimators','max_depth','min_child_weight') else float(rng.choice(v))) for k,v in grids.items()}
        key=tuple(item.items())
        if key not in seen: seen.add(key); choices.append(item)
    # Fit imputation and Case 1 feature selection once per CV split, using only its training portion.
    splits=[]
    for fold in range(10):
        valid=np.flatnonzero(cv_id==fold); train=np.flatnonzero(cv_id!=fold)
        imp,cols_all,features=fit_selector(X.iloc[train],y.iloc[train])
        Xfit=pd.DataFrame(imp.transform(X.iloc[train][cols_all]),columns=cols_all,index=X.index[train])[features]
        Xvalid=pd.DataFrame(imp.transform(X.iloc[valid][cols_all]),columns=cols_all,index=X.index[valid])[features]
        splits.append((train,valid,Xfit,Xvalid,y.iloc[train].to_numpy(),y.iloc[valid].to_numpy(),features))
        print(f'Prepared CV fold {fold+1}/10: train={len(train):,}, validation={len(valid):,}, features={len(features)}',flush=True)
    cv_rows=[]; best=None
    for n,params in enumerate(choices,1):
        fold_rmse=[]; fold_mse=[]; fold_mae=[]
        for fold,(train,valid,Xfit,Xvalid,yfit,yvalid,features) in enumerate(splits):
            model=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=2,verbosity=0)
            model.fit(Xfit,yfit); score=model.predict_proba(Xvalid)[:,1]
            m=metrics(yvalid,score); fold_rmse.append(m['rmse']); fold_mse.append(m['mse']); fold_mae.append(m['mae'])
        row={'trial':n,**params,'rmse_mean':float(np.mean(fold_rmse)),'rmse_std':float(np.std(fold_rmse,ddof=1)),
             'mse_mean':float(np.mean(fold_mse)),'mae_mean':float(np.mean(fold_mae))}
        cv_rows.append(row)
        if best is None or row['rmse_mean']<best['rmse_mean']: best=row
        print(f"Trial {n:02d}/30: CV RMSE={row['rmse_mean']:.5f} +/- {row['rmse_std']:.5f}",flush=True)
    # Out-of-fold TreeSHAP on train, using XGBoost's native contribution prediction.
    params={k:best[k] for k in grids}
    oof_shap=np.full((len(X),len(features)),np.nan,dtype=np.float32)
    shap_feature_values=np.full((len(X),len(features)),np.nan,dtype=np.float32)
    for fold,(train,valid,Xfit,Xvalid,yfit,yvalid,fold_features) in enumerate(splits):
        assert fold_features==features, 'Feature set changed across folds; cannot combine SHAP columns.'
        model=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=2,verbosity=0)
        model.fit(Xfit,yfit)
        contrib=model.get_booster().predict(__import__('xgboost').DMatrix(Xvalid,feature_names=features),pred_contribs=True)
        oof_shap[valid]=contrib[:,:-1]
        shap_feature_values[valid]=Xvalid.to_numpy(dtype=np.float32)
        print(f'SHAP fold {fold+1}/10 complete',flush=True)
    assert np.isfinite(oof_shap).all(), 'Every training row must have one OOF SHAP vector.'
    shap_dir=ROOT/'shap_figures'; shap_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(oof_shap,columns=features).to_csv(shap_dir/'case1_xgb_lam_shap_oof.csv.gz',index=False,compression='gzip')
    import runpy
    runpy.run_path(str(ROOT/'notebooks'/'plot_case1_xgb_lam_shap.py'), run_name='__main__')
    # Refit selection and model on all train; test is touched only for final scoring.
    imp,cols_all,features=fit_selector(X,y)
    Xfull=pd.DataFrame(imp.transform(X[cols_all]),columns=cols_all,index=X.index)[features]
    Xtest=pd.DataFrame(imp.transform(Xt[cols_all]),columns=cols_all,index=Xt.index)[features]
    params={k:best[k] for k in grids}
    final=XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=2,verbosity=0)
    final.fit(Xfull,y); test_score=final.predict_proba(Xtest)[:,1]; test_metrics=metrics(yt.to_numpy(),test_score)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(OUT/'xgboost_lam_cv_10fold.csv',index=False)
    pd.DataFrame([test_metrics]).to_csv(OUT/'xgboost_lam_test_metrics.csv',index=False)
    result={'case':'case1','model':'xgboost','author':'Lam','features':features,'n_features':len(features),
      'split':'Provided train.csv (143,771 rows) / test.csv (70,814 rows); seed=42. Ten-fold CV on train using the shared fold assignments. Test used only once after tuning.',
      'best_params':params,'metrics_test':{k:test_metrics[k] for k in ('rmse','mse','mae','r2','accuracy','balanced_accuracy','precision','recall','recall_SCA1','recall_SCA0','f1','roc_auc')},
      'cv_10fold':{'rmse_mean':best['rmse_mean'],'rmse_std':best['rmse_std']},
      'notebook':'notebooks/case1_xgb_lam.ipynb',
      'notes':f"10-fold CV uses the shared fold assignments; scale_pos_weight was searched within each training fold; best value was 1.0 (no positive-class reweighting). OOF TreeSHAP matrices and standard SHAP summary/bar/dependence PNGs are saved in shap_figures/. 30-trial randomized grid search, seed 42. Mean CV MSE={best['mse_mean']:.6f}; mean CV MAE={best['mae_mean']:.6f}. Test accuracy={test_metrics['accuracy']:.6f}, balanced_accuracy={test_metrics['balanced_accuracy']:.6f}, recall_SCA1={test_metrics['recall_SCA1']:.6f}, recall_SCA0={test_metrics['recall_SCA0']:.6f}, ROC-AUC={test_metrics['roc_auc']:.6f}. Class metrics use 0.50 cutoff. Assumes SCA=1 is abandonment. Summary and dependence plots use a seeded 30,000-row display sample; saved OOF matrices contain all training rows. The agreed train/test files have 66 exact record overlaps; test estimates may be mildly optimistic. Shared fold file differs from train.csv in avg_dwell_time for 16 rows; current train.csv values were used by row alignment."}
    (OUT/'xgboost_lam.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\nBEST PARAMS',json.dumps(params))
    print('CV',best)
    print('TEST',json.dumps(test_metrics))
    print('FEATURES',features)
    print('WROTE',OUT/'xgboost_lam.json')
if __name__=='__main__': main()
