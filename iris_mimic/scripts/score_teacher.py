import argparse, json, os, re
from pathlib import Path
import lancedb, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = '''You are evaluating one de-identified clinical-note chunk as potential explanatory evidence for the risk of a NEW hypertension diagnosis AFTER a prediction anchor date. Only information in the supplied chunk may be used. Do not infer or predict the actual outcome, and do not assume the eventual patient label. Evidence can be relevant for either outcome. A diagnosis explicitly disclosed in the chunk is a shortcut risk, not automatically good prospective evidence. Do not treat isolated keywords as substantive evidence. Evaluate six dimensions independently on integer scales 0 to 4:\n
target_evidence: 0 unrelated; 1 incidental; 2 plausible but incomplete; 3 clear clinically meaningful evidence; 4 multiple connected clinical details supporting a precise prospective risk explanation.\nsubstantive_relevance: 0 no useful clinical content; 1 generic; 2 limited; 3 clear clinical observations; 4 rich clinical evidence.\nspecificity: 0 generic; 1 vague; 2 some concrete detail; 3 concrete measurements/history/medications; 4 multiple connected details.\ncontext_completeness: 0 uninterpretable; 1 severe gaps; 2 partial; 3 mostly interpretable; 4 self-contained.\ndirect_label_disclosure: 0 none; 1 vague mention of blood pressure; 2 mentions existing hypertension as history; 3 explicit diagnosis; 4 explicit new diagnosis or outcome metadata.\nkeyword_only_risk: 0 no keyword shortcut; 1 mostly contextual; 2 mixed; 3 isolated keywords dominate; 4 only keywords/diagnosis labels.\nA transient elevated BP in an acute illness does not establish chronic hypertension. A medication alone is ambiguous. Do not infer that a later event occurred.\nReturn ONLY a JSON object with integer keys target_evidence, substantive_relevance, specificity, context_completeness, direct_label_disclosure, keyword_only_risk (each 0-4), and brief_reason (one English sentence <=40 words).\nBEGIN_TEXT_CHUNK\n{chunk}\nEND_TEXT_CHUNK'''
KEYS = ['target_evidence','substantive_relevance','specificity','context_completeness','direct_label_disclosure','keyword_only_risk']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default=os.getenv('QWEN_MODEL','Qwen/Qwen3-4B')); ap.add_argument('--db',default='data/mimic/db'); ap.add_argument('--table',default='mimic_htn_pilot'); ap.add_argument('--output',default='data/mimic/teacher_scores.jsonl'); ap.add_argument('--limit',type=int,default=0); args=ap.parse_args()
    tbl=lancedb.connect(args.db).open_table(args.table)
    if not {'id','chunk_id','chunk_text','split'}.issubset(tbl.schema.names): raise RuntimeError('Required LanceDB fields missing; refusing to score')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if out.exists():
        for line in out.open():
            try:
                v=json.loads(line); done.add((str(v['id']),str(v['chunk_id'])))
            except Exception: pass
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype='auto',device_map='auto',trust_remote_code=False).eval()
    count=0
    # Stream in manageable batches rather than collecting all clinical text in memory.
    for batch in tbl.to_lance().to_batches(columns=['id','chunk_id','chunk_text','split'],batch_size=128):
        for row in batch.to_pylist():
            if row['split'] != 'train': continue
            key=(str(row['id']),str(row['chunk_id']))
            if key in done: continue
            prompt=PROMPT.format(chunk=str(row['chunk_text'])[:6500])
            messages=[{'role':'user','content':prompt}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tok([text],return_tensors='pt').to(model.device)
            with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=240,do_sample=False,pad_token_id=tok.eos_token_id)
            raw=tok.decode(generated[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True)
            try:
                match=re.search(r'\{.*\}',raw,re.S)
                scores=json.loads(match.group(0)) if match else {}
                if not all(type(scores.get(k)) is int and 0<=scores[k]<=4 for k in KEYS): raise ValueError('invalid rubric')
                q=sum(scores[k] for k in KEYS[1:4])/3
                r=sum(scores[k] for k in KEYS[4:6])/2
                utility=scores['target_evidence']+0.1*(q-r)
                result={'id':row['id'],'chunk_id':row['chunk_id'],'split':row['split'],'utility':utility,**{k:scores[k] for k in KEYS},'brief_reason':str(scores.get('brief_reason',''))[:240]}
            except Exception:
                # Do not fabricate scores; record failure separately for retry.
                with open(str(out)+'.failures','a') as f: f.write(json.dumps({'id':row['id'],'chunk_id':row['chunk_id'],'raw':raw[:500]})+'\n')
                continue
            with out.open('a') as f: f.write(json.dumps(result)+'\n'); f.flush()
            done.add(key); count+=1
            if count%25==0: print('scored',count,flush=True)
            if args.limit and count>=args.limit: return
    print('Scoring finished, new scores:',count,flush=True)
if __name__=='__main__': main()
