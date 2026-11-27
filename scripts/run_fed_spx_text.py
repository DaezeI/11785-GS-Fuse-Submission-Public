#!/usr/bin/env python3
"""Exact #55 preparation: frozen full-precision Llama-3.1-8B and frozen CNN.
Only train/validation are opened. Missing frozen inputs are fatal before GPU loading.
"""
import argparse, hashlib, json, os, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.fed_spx_endpoints import load_endpoint_cohorts, HORIZONS
from model.compact_return import CompactReturn
from scripts.run_fed_spx_endpoints_price import endpoint_metrics, bootstrap_gain

EXCLUDE={'stmt_20081029a','stmt_20140430a'}
EXPECTED={'train':'7d34c2a1f5a34ded2bc459cf1894f390048b2cb760479cd5475dc70aa7fc5760','val':'b2e52208741d3b1a4f32d55e5cd06292b275eaa92028b81b1d4aff8abd0495be'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def read_texts(folder):
 result={}
 for split in ('train','val'):
  p=folder/(split+'.jsonl')
  if sha(p)!=EXPECTED[split]: raise ValueError('JSONL fingerprint mismatch: '+split)
  rows=[json.loads(s) for s in p.read_text().splitlines()]
  assert len(rows)=={'train':173,'val':149}[split]
  ids=set()
  for r in rows:
   assert r['split']==split and r['event_id'] not in ids
   ids.add(r['event_id'])
   assert hashlib.sha256(r['model_text'].encode()).hexdigest()==r['model_text_sha256']
  result[split]=[r for r in rows if r['event_id'] not in EXCLUDE]
 return result

def frozen_arm(checkpoint,text_dim,fusion='add',joint_hidden_layers=1):
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 base=CompactReturn(output_dim=2).cuda().eval();base.load_state_dict(state['state_dict'])
 arm=CompactReturn(text_dim=text_dim,output_dim=2,fusion=fusion,joint_hidden_layers=joint_hidden_layers).cuda()
 missing,unexpected=arm.load_state_dict(state['state_dict'],strict=False)
 assert not unexpected and all(k.startswith('text_head.') for k in missing)
 arm.freeze_price()
 return base,arm,state

def donors(rows,seed):
 rng=np.random.default_rng(seed); times=pd.to_datetime([r['text_cutoff_utc'] for r in rows],utc=True)
 out=np.full(len(rows),-1,dtype=int)
 for i,r in enumerate(rows):
  eligible=[j for j,s in enumerate(rows) if j!=i and s['market_et']!=r['market_et'] and times[j]<=times[i]]
  if eligible: out[i]=rng.choice(eligible)
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--text-dir',type=Path,required=True);p.add_argument('--price-csv',type=Path,required=True)
 p.add_argument('--frozen-dir',type=Path,required=True);p.add_argument('--llama-dir',type=Path,required=True)
 p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--max-tokens',type=int,default=2048)
 p.add_argument('--epochs',type=int,default=30);p.add_argument('--preflight-only',action='store_true')
 p.add_argument('--fusion',choices=('add','joint_mlp'),default='add')
 p.add_argument('--joint-hidden-layers',type=int,choices=(1,2),default=1)
 a=p.parse_args(); missing=[]
 fusion_id='Add' if a.fusion=='add' else f'MLP{a.joint_hidden_layers}-Residual'
 architecture=f'Llama-3.1-8B(frozen)+CNN(frozen)~Pool~{fusion_id}'
 for f in [a.price_csv,a.frozen_dir/'metrics.json',a.frozen_dir/'val-paired.npz',a.llama_dir/'config.json',*[a.frozen_dir/f'compact-price-seed-{s}.pt' for s in (0,1,2)]]:
  if not f.is_file():missing.append(str(f))
 if missing: raise SystemExit('Missing required exact #55 artifacts:\n'+'\n'.join(missing))
 texts=read_texts(a.text_dir)
 config=json.loads((a.llama_dir/'config.json').read_text())
 assert config['hidden_size']==4096 and config['num_hidden_layers']==32 and config['num_attention_heads']==32
 published=json.loads((ROOT/'results/fed-spx-endpoints-price-v1/metrics.json').read_text())
 for r in published['price_only_seeds']:
  assert sha(a.frozen_dir/f"compact-price-seed-{r['seed']}.pt")==r['checkpoint_sha256']
 cohorts,audit=load_endpoint_cohorts(a.price_csv)
 joined={}
 for split in ('train','val'):
  c=cohorts[split]; index=dict(zip(c.rows.event_id,range(len(c.rows))))
  assert all(r['event_id'] in index for r in texts[split])
  ix=[index[r['event_id']] for r in texts[split]]
  rows=c.rows.iloc[ix].reset_index(drop=True)
  for i,r in enumerate(texts[split]):
   assert pd.Timestamp(r['text_cutoff_utc'])==pd.Timestamp(rows.iloc[i].text_cutoff_utc)
  weights=1/rows.groupby('market_et').event_id.transform('size').to_numpy(float)
  joined[split]=(c.history[ix],c.returns[ix],weights,rows)
 assert (len(texts['train']),len(texts['val']))==(171,149)
 print('Exact input preflight passed: 171 train / 149 val; test not opened',flush=True)
 print('Architecture: '+architecture,flush=True)
 if a.preflight_only:return
 assert torch.cuda.is_available()
 torch.use_deterministic_algorithms(True)
 from transformers import AutoModel,AutoTokenizer
 a.out_dir.mkdir(parents=True,exist_ok=False)
 tokenizer=AutoTokenizer.from_pretrained(a.llama_dir,local_files_only=True)
 encoder=AutoModel.from_pretrained(a.llama_dir,torch_dtype=torch.float32,local_files_only=True).cuda().eval()
 for param in encoder.parameters():param.requires_grad_(False)
 features={};truncated={}
 with torch.inference_mode():
  for split in ('train','val'):
   vectors=[]; lengths=[]
   for i,r in enumerate(texts[split]):
    encoded=tokenizer(r['model_text'],return_tensors='pt',truncation=True,max_length=a.max_tokens)
    lengths.append(int(encoded.input_ids.shape[1]));encoded={k:v.cuda() for k,v in encoded.items()}
    hidden=encoder(**encoded,use_cache=False).last_hidden_state
    mask=encoded['attention_mask'].unsqueeze(-1)
    vectors.append(((hidden*mask).sum(1)/mask.sum(1)).cpu().numpy()[0])
    if i%25==0:print(f'{split} embedding {i+1}/{len(texts[split])}',flush=True)
   features[split]=np.stack(vectors);truncated[split]=sum(n==a.max_tokens for n in lengths)
 np.savez_compressed(a.out_dir/'features.npz',**features)
 del encoder;torch.cuda.empty_cache()
 mean=features['train'].mean(0);std=np.maximum(features['train'].std(0),1e-6)
 z={k:torch.as_tensor((v-mean)/std,dtype=torch.float32,device='cuda') for k,v in features.items()}
 np.savez_compressed(a.out_dir/'text-scaler.npz',mean=mean,std=std)
 reports=[]; predictions=[]; baseline_predictions=[];zero_predictions=[];shuffle_predictions=[]
 vr=joined['val'][3]; dates=vr.market_et.str[:10].to_numpy(str);w=joined['val'][2]; truth=joined['val'][1]
 donor_draws={seed:donors(texts['val'],seed) for seed in (11,23,37)}
 donor_ok=np.logical_and.reduce([d>=0 for d in donor_draws.values()])
 np.savez_compressed(a.out_dir/'donors.npz',**{str(k):v for k,v in donor_draws.items()})
 for seed in (0,1,2):
  torch.manual_seed(seed)
  base,arm,state=frozen_arm(a.frozen_dir/f'compact-price-seed-{seed}.pt',4096,a.fusion,a.joint_hidden_layers)
  scale=np.asarray(state['target_scale'],dtype=np.float64)
  x={k:torch.as_tensor(v[0]/state['input_scale'],dtype=torch.float32,device='cuda') for k,v in joined.items()}
  y=torch.as_tensor(joined['train'][1]/scale,dtype=torch.float32,device='cuda')
  tw=torch.as_tensor(joined['train'][2]/joined['train'][2].mean(),dtype=torch.float32,device='cuda')
  with torch.inference_mode():
   baseline=base(x['val']);assert torch.equal(arm(x['val']),baseline)
   assert torch.equal(arm(x['val'],z['val']),baseline)
  original={k:v.detach().clone() for k,v in arm.state_dict().items() if not k.startswith('text_head.')}
  opt=torch.optim.AdamW(arm.text_head.parameters(),lr=.001,weight_decay=.001)
  generator=torch.Generator().manual_seed(seed); best=float('inf'); trace=[];train_trace=[];step_trace=[]
  for epoch in range(a.epochs+1):
   if epoch:
    arm.train()
    for ix in torch.randperm(len(y),generator=generator).split(64):
     ix=ix.cuda();opt.zero_grad(set_to_none=True)
     err=torch.nn.functional.smooth_l1_loss(arm(x['train'][ix],z['train'][ix]),y[ix],reduction='none')
     loss=(err.mean(1)*tw[ix]).mean();assert torch.isfinite(loss);loss.backward()
     norm=torch.nn.utils.clip_grad_norm_(arm.text_head.parameters(),1.,error_if_nonfinite=True);opt.step()
     step_trace.append({'update':len(step_trace)+1,'epoch':epoch,'minibatch_loss':loss.item(),'gradient_norm':float(norm)})
   arm.eval()
   with torch.inference_mode():
    pred=arm(x['val'],z['val']).cpu().numpy().astype(np.float64)*scale
    train_err=torch.nn.functional.smooth_l1_loss(arm(x['train'],z['train']),y,reduction='none')
    train_trace.append(float((train_err.mean(1)*tw).mean()))
   score=float(np.average(np.abs(pred[:,0]-truth[:,0]),weights=w));trace.append(score)
   if score<best:
    best=score; chosen=epoch; saved={k:v.detach().cpu().clone() for k,v in arm.state_dict().items()}
  torch.save({'state_dict':arm.state_dict(),'optimizer_state_dict':opt.state_dict(),'order_generator_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),'epoch':a.epochs,'updates':len(step_trace),'architecture':architecture},a.out_dir/f'last-training-state-{seed}.pt')
  pd.DataFrame(step_trace).to_csv(a.out_dir/f'optimizer-steps-{seed}.csv',index=False)
  pd.DataFrame({'epoch':range(a.epochs+1),'train_smooth_l1':train_trace,'val_mae20_bp':trace}).to_csv(a.out_dir/f'learning-curves-{seed}.csv',index=False)
  arm.load_state_dict(saved);arm.eval()
  assert all(torch.equal(arm.state_dict()[k],v) for k,v in original.items())
  with torch.inference_mode():
   assert torch.equal(arm(x['val']),baseline)
   real=arm(x['val'],z['val']).cpu().numpy().astype(np.float64)*scale
   zero=arm(x['val'],torch.zeros_like(z['val'])).cpu().numpy().astype(np.float64)*scale
   draws=[]
   for d in donor_draws.values():
    safe=np.maximum(d,0);draws.append(arm(x['val'],z['val'][safe]).cpu().numpy().astype(np.float64)*scale)
  predictions.append(real);baseline_predictions.append(baseline.cpu().numpy().astype(np.float64)*scale)
  zero_predictions.append(zero);shuffle_predictions.append(draws)
  checkpoint=a.out_dir/f'text-seed-{seed}.pt';torch.save({'state_dict':saved,'architecture':architecture,'fusion':a.fusion,'joint_hidden_layers':a.joint_hidden_layers if a.fusion=='joint_mlp' else None,'selected_epoch':chosen,'text_mean':mean,'text_std':std,'price_state':{k:v for k,v in state.items() if k!='state_dict'}},checkpoint)
  reports.append({'seed':seed,'selected_epoch':chosen,'train_smooth_l1_trace':train_trace,'val_mae20_trace':trace,'optimizer_updates':len(step_trace),'checkpoint_sha256':sha(checkpoint)})
  print(f'Seed {seed}: selected epoch {chosen}, MAE20 {best:.4f} bp',flush=True)
  del base,arm
 real=np.mean(predictions,axis=0);base=np.mean(baseline_predictions,axis=0);zero=np.mean(zero_predictions,axis=0)
 shuffled=np.mean(shuffle_predictions,axis=(0,1));boot={'bootstrap_seed':23,'bootstrap_replicates':2000}
 scores={'real':endpoint_metrics(truth,real,w),'text_zero':endpoint_metrics(truth,zero,w),'price_baseline':endpoint_metrics(truth,base,w),'Persistence':endpoint_metrics(truth,np.zeros_like(truth),w)}
 paired={}
 for name,other in [('price_baseline',base),('text_zero',zero),('text_shuffled',shuffled)]:
  keep=donor_ok if name=='text_shuffled' else np.ones(len(w),bool)
  gain=np.abs(other[keep]-truth[keep])-np.abs(real[keep]-truth[keep])
  paired[name]={'rows':int(keep.sum()),'gain_bp':np.average(gain,weights=w[keep],axis=0).tolist(),'ci95':bootstrap_gain(gain,dates[keep],w[keep],boot)}
 np.savez_compressed(a.out_dir/'val-paired.npz',architecture=architecture,event_ids=vr.event_id.to_numpy(str),truth=truth,real=predictions,price=baseline_predictions,text_zero=zero_predictions,text_shuffled=shuffle_predictions,donor_eligible=donor_ok,weights=w)
 report={'architecture':architecture,'fusion':a.fusion,'joint_hidden_layers':a.joint_hidden_layers if a.fusion=='joint_mlp' else None,'train_cohort':'FED-PXTX-ALL-REL5-v2 minus 2 truncated train texts','price_train_cohort':'FED-PX-REL5-v1','score_cohort':'FED-PXTX-ALL-REL5-v2 val','train_rows':171,'val_rows':149,'test_loaded':False,'max_tokens':a.max_tokens,'precision':'float32, no quantization','pooling':'attention-mask mean of final hidden state, including special tokens','selection':f'{a.epochs} fixed epochs plus epoch 0, choose validation MAE20; exploratory','truncation_at_cap':truncated,'price_audit':audit,'seeds':reports,'metrics':scores,'paired':paired,'text_off_exact':True,'frozen_price_unchanged':True,'torch':torch.__version__,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(__file__),'llama_config_sha256':sha(a.llama_dir/'config.json'),'input_jsonl_sha256':EXPECTED,'frozen_checkpoint_sha256':{str(seed):sha(a.frozen_dir/f'compact-price-seed-{seed}.pt') for seed in (0,1,2)},'limitations':['Validation was already used for price checkpoint selection.','ALL includes inferred evidence classes.','Shuffled comparison excludes rows without an eligible past donor.','2048-token truncation may omit later content.']}
 (a.out_dir/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print('TRAINING_AND_VALIDATION_COMPLETE; test reserved',flush=True)
if __name__=='__main__':main()
