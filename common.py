
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
def binary_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    tp=int(np.sum((y==1)&(p==1))); fp=int(np.sum((y==0)&(p==1)))
    fn=int(np.sum((y==1)&(p==0))); tn=int(np.sum((y==0)&(p==0)))
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    return dict(accuracy=float(np.mean(y==p)),precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,confusion_matrix={'tn':tn,'fp':fp,'fn':fn,'tp':tp})
def save_report(report):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows=''.join('<tr><th>'+str(k)+'</th><td><pre>'+json.dumps(v,indent=2)+'</pre></td></tr>' for k,v in report.items())
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Experiment report</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#f3f6fc;color:#173254}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:15px;text-align:left}pre{white-space:pre-wrap}th{width:25%}</style><h1>Reproducible experiment report</h1><p>Synthetic data; educational results, not production evidence.</p><table>'+rows+'</table>',encoding='utf-8')
    print(json.dumps(report,indent=2))
