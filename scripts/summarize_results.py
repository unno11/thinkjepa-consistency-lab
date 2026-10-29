"""Recompute published descriptive summaries from archived validation metrics.

No model, dataset, checkpoint, GPU or trajectory prediction is loaded here.
"""
from pathlib import Path
import csv,json,math,statistics,hashlib,argparse

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(42,43,44)
ROLES=('B','KProj','MG')
RATIOS={'VelocityRatio','AccelerationRatio'}

def summarize(write=False):
    provenance=json.loads((ROOT/'provenance/source_snapshot.json').read_text(encoding='utf-8'))
    for item in provenance['result_files']:
        actual=hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()
        if actual!=item['source_sha256']:raise ValueError('Archived evidence changed: '+item['path'])
    tables={}
    for seed in SEEDS:
        for role in (*ROLES,'B_repeated'):
            path=ROOT/'results/per_sample'/f'{role}_seed{seed}.csv'
            with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            table={r['sample']:r for r in rows}
            if len(rows)!=200 or len(table)!=200:raise ValueError('Expected 200 unique validation samples')
            tables[(seed,role)]=table
        if any(set(tables[(seed,role)])!=set(tables[(42,'B')]) for role in (*ROLES,'B_repeated')):raise ValueError('Sample membership mismatch')
    metrics=[k for k in next(iter(tables[(42,'B')].values())) if k not in ('sample','role')]
    max_delta=0.0
    for seed in SEEDS:
        for sample,row in tables[(seed,'B')].items():
            for metric in metrics:max_delta=max(max_delta,abs(float(row[metric])-float(tables[(seed,'B_repeated')][sample][metric])))
    if max_delta>1e-10:raise ValueError('Repeated B equivalence failed')
    output=[]
    for role in ROLES:
        for metric in metrics:
            means=[]
            for seed in SEEDS:
                values=[float(row[metric]) for row in tables[(seed,role)].values()]
                if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite metric')
                if metric in RATIOS:
                    if any(v<=0 for v in values):raise ValueError('Nonpositive dynamic ratio')
                    values=[abs(math.log(v)) for v in values]
                means.append(statistics.mean(values))
            output.append({'role':role,'metric':metric,'transform':'abs_log_ratio' if metric in RATIOS else 'identity','n_training_seeds':3,'samples_per_seed':200,'mean':statistics.mean(means),'seed_sd':statistics.stdev(means)})
    with (ROOT/'results/model_seed_summary.csv').open(newline='',encoding='utf-8') as f:archived=list(csv.DictReader(f))
    lookup={(r['role'],r['metric']):r for r in output}
    max_summary_delta=0.0
    for row in archived:
        contrast=row['contrast'];candidate,baseline={'KProj_minus_B':('KProj','B'),'MG_minus_B':('MG','B'),'MG_minus_KProj':('MG','KProj')}[contrast]
        for role,prefix in [(baseline,'baseline'),(candidate,'candidate')]:
            for field,suffix in [('mean','seed_mean'),('seed_sd','seed_sd')]:
                max_summary_delta=max(max_summary_delta,abs(lookup[(role,row['metric'])][field]-float(row[prefix+'_'+suffix])))
    if max_summary_delta>1e-9:raise ValueError('Archived summary does not match per-sample evidence')
    tasks={sample.split('/')[-2] for sample in tables[(42,'B')]}
    report={'training_seeds':list(SEEDS),'validation_samples_per_seed':200,'tasks':len(tasks),'repeated_baseline_max_delta':max_delta,'archived_summary_max_delta':max_summary_delta,'final_test_evaluated':False,'scope':'Recomputed archived per-sample metrics; no training or inference rerun'}
    if write:
        with (ROOT/'results/summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
        (ROOT/'results/verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report,output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    report,rows=summarize(args.write)
    print(json.dumps(report,indent=2))
    print('\nMetric                     B             KProj         MG')
    for metric in ['ADE_m','FDE_m','BoneErrorAll_mm','AngleError_deg','VelocityRatio','AccelerationRatio']:
        values=[next(r for r in rows if r['role']==role and r['metric']==metric) for role in ROLES]
        print(f'{metric:24s} '+ '  '.join(f'{r["mean"]:.6f}' for r in values))
