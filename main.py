"""Compare numeric production features with a fixed reference using PSI and KS."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ks_2samp
def psi(reference,current,bins=10):
 edges=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)))
 if len(edges)<3:return 0.0
 edges[0],edges[-1]=-np.inf,np.inf
 p=np.histogram(reference,bins=edges)[0]/len(reference);q=np.histogram(current,bins=edges)[0]/len(current);eps=1e-6
 p=np.clip(p,eps,None);q=np.clip(q,eps,None)
 return float(np.sum((q-p)*np.log(q/p)))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--reference",required=True);p.add_argument("--current",required=True);p.add_argument("--columns",default=None,help="Comma-separated numeric columns; default is numeric intersection");p.add_argument("--psi-threshold",type=float,default=.2);p.add_argument("--out",default="outputs");a=p.parse_args()
 ref=pd.read_csv(a.reference);cur=pd.read_csv(a.current);cols=a.columns.split(",") if a.columns else sorted(set(ref.select_dtypes(include="number"))&set(cur.select_dtypes(include="number")))
 report={"reference_rows":len(ref),"current_rows":len(cur),"features":{}}
 for col in cols:
  x=pd.to_numeric(ref[col],errors="coerce").dropna().to_numpy();y=pd.to_numeric(cur[col],errors="coerce").dropna().to_numpy()
  if len(x)<20 or len(y)<20:continue
  score=psi(x,y);ks=ks_2samp(x,y)
  report["features"][col]={"psi":score,"ks_statistic":float(ks.statistic),"ks_pvalue":float(ks.pvalue),"reference_missing_rate":float(ref[col].isna().mean()),"current_missing_rate":float(cur[col].isna().mean()),"alert":score>=a.psi_threshold}
 report["psi_threshold"]=a.psi_threshold;report["drifted_features"]=[k for k,v in report["features"].items() if v["alert"]]
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"drift_report.json").write_text(json.dumps(report,indent=2));pd.DataFrame([{"feature":k,**v} for k,v in report["features"].items()]).to_csv(out/"drift_report.csv",index=False);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
