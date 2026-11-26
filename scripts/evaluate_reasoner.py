"""Frozen diagnostic pairing of SJTU chat and reasoner; no robot motion.

Default execution is a dry run. --live sends only frozen prompts and synthetic
scene/text inputs. Existing responses are never replaced or retried for quality.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import time
from unittest.mock import patch

import httpx
import numpy as np

from arm_lab.model_access import DEEPSEEK_ENDPOINT, ModelAccess, ModelFailure, campus_api_key, planning_task
from arm_lab.planner_scoring import has_motion, score_response
from scripts.evaluate_prompt_trial import exclusive, save
from scripts.replay_planner100_guard import intent_document

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/safety/reasoner/paired.json'
CASE_IDS = ('P003','P008','P010','P011','P013','P017','P055','P058','P059','P066',
            'P070','P075','P078','P081','P029','P033','P035','P085','P086','P088')
MODELS = ('deepseek-reasoner', 'deepseek-chat')
MAX_ATTEMPTS = 48
MAX_RETRIES = 2


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def freeze(reasoner_tokens=4096, prior_attempts=0, max_attempts=48, interval=6.2, timeout=120):
    if not 512 <= reasoner_tokens <= 8192:
        raise ValueError('Reasoner budget must be 512..8192 tokens.')
    if not 0 <= prior_attempts < max_attempts <= MAX_ATTEMPTS or max_attempts-prior_attempts < 44:
        raise ValueError('Cumulative budget must leave 44 attempts for the planned pairs, and cannot exceed 48.')
    if interval < 6.2 or not 30 <= timeout <= 180:
        raise ValueError('Use at least 6.2 seconds between starts and a 30..180 second request timeout.')
    data_bytes = (ROOT/'configs/planner100.json').read_bytes()
    lookup = {row['id']:row for row in json.loads(data_bytes)['cases']}
    cases = [lookup[id_] for id_ in CASE_IDS]
    speech_bytes = (ROOT/'results/safety/sensevoice/flow_live.json').read_bytes()
    speech = json.loads(speech_bytes)
    transfer = next(r for r in speech['rows'] if r['id']=='transfer')
    transcript = transfer['asr_response']['text']
    if '右干' not in transcript or transfer['command']['text'] != transcript:
        raise ValueError('The saved raw SenseVoice source is not the expected unedited diagnostic.')
    for slot,xy in [('A',[215,77]),('C',[245,0])]:
        cases.append({'id':'ASR_'+slot, 'split':'diagnostic_reused_transcription', 'family':'right_gan',
                      'command':transcript, 'scene':{'objects':[{'id':'red_1','slot':slot,'xy_mm':xy}]},
                      'route':'planner', 'expected':{'steps':[{'action':'transfer','object':'red_1','from':slot,'to':'B'}]}})
    if sum('steps' in c['expected'] for c in cases) != 16 or len(cases)!=22:
        raise ValueError('Diagnostic composition changed.')
    prompt = (ROOT/'configs/planner100_finalist.txt').read_text(encoding='utf-8').strip()
    plan = [{'case_id':case['id'],'model':model} for i,case in enumerate(cases)
            for model in (MODELS if i%2==0 else tuple(reversed(MODELS)))]
    return {'schema_version':1,'prompt':prompt,'prompt_sha256':digest(prompt.encode()),
            'source_dataset_sha256':digest(data_bytes),'source_speech_sha256':digest(speech_bytes),
            'cases':cases,'schedule':plan,'model_tokens':{'deepseek-chat':160,'deepseek-reasoner':reasoner_tokens},
            'prior_campaign_attempts':prior_attempts,'max_cumulative_attempts':max_attempts,
            'minimum_interval_seconds':interval,'request_timeout_seconds':timeout,
            'max_automatic_retries':MAX_RETRIES,
            'scope':'Selected, previously inspected diagnostics, not a new blind test. The two ASR cases reuse one saved transcript. Raw planner is tested directly; no physical execution. Missing units are handled locally in production and tested by the full chain.'}


def payload(manifest, case, model):
    return {'model':model,'temperature':0,'max_tokens':manifest['model_tokens'][model],'stream':False,
            'messages':[{'role':'system','content':manifest['prompt']},
                        {'role':'user','content':json.dumps(planning_task(case['command'],case['scene']),ensure_ascii=False)}]}


def guard_result(case, row):
    """Replay only the content through production validation, without HTTP or motion."""
    if row.get('http_status') != 200 or row.get('finish_reason') != 'stop':
        return {'outcome':'unavailable','wrong_motion_admitted':False}
    text = row.get('raw')
    if not isinstance(text,str):
        return {'outcome':'rejected','reason':'missing_final_content','wrong_motion_admitted':False}
    text = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]+?)\s*```',text,re.I)
    try:
        parsed = json.loads(fence[1] if fence else text)
    except ValueError:
        return {'outcome':'rejected','reason':'invalid_json','wrong_motion_admitted':False}
    access = ModelAccess()
    try:
        with patch('arm_lab.model_access.campus_api_key',return_value='offline-placeholder'), \
                patch.object(access,'chat_json',return_value=parsed):
            accepted = intent_document(access.plan(case['command'],case['scene'],approved_once=True))
    except (ModelFailure,ValueError,TypeError) as exc:
        return {'outcome':'rejected','reason':str(exc),'wrong_motion_admitted':False}
    after = score_response(case,json.dumps(accepted,ensure_ascii=False))
    motion = has_motion(accepted)
    return {'outcome':'admitted_motion' if motion else 'ask_or_nonmotion','accepted_plan':accepted,
            'goal_matches_reference':after['semantic_correct'],
            'wrong_motion_admitted':motion and not after['semantic_correct']}


def summarize(book):
    manifest=book['manifest']
    lookup={c['id']:c for c in manifest['cases']}
    rows=[]
    for attempt in book['attempts']:
        if attempt['state']!='answered':
            continue
        case=lookup[attempt['case_id']]
        rows.append({'case_id':case['id'],'model':attempt['model'],
                     'kind':'action' if 'steps' in case['expected'] else 'ask',
                     'scores':score_response(case,attempt.get('raw'),http_status=attempt.get('http_status'),
                                             finish_reason=attempt.get('finish_reason')),
                     'guard':guard_result(case,attempt),'attempt_index':attempt['index']})
    summary={'cumulative_attempts':manifest['prior_campaign_attempts']+len(book['attempts']),
             'this_report_attempts':len(book['attempts']),'models':{},'paired':{},
             'automatic_retries':sum(a.get('is_retry',False) for a in book['attempts'])}
    for model in MODELS:
        group=[r for r in rows if r['model']==model]
        attempts=[a for a in book['attempts'] if a['model']==model]
        answered=[a for a in attempts if a['state']=='answered']
        times=[a['seconds'] for a in answered]
        totals=Counter()
        for a in answered:
            for key,value in (a.get('usage') or {}).items():
                if type(value) in (int,float):
                    totals[key]+=value
        summary['models'][model]={
            'planned':22,'answered':len(group),'attempts':len(attempts),
            'schema_valid':sum(r['scores']['schema_valid'] for r in group),
            'wrong_physical_proposals':sum(r['scores']['dangerous_motion'] for r in group),
            'wrong_motion_admitted_by_guard':sum(r['guard']['wrong_motion_admitted'] for r in group),
            'truncated':sum(a.get('finish_reason')=='length' for a in answered),
            'reasoning_present':sum(a.get('reasoning_characters',0)>0 for a in answered),
            'seconds_median':float(np.median(times)) if times else None,
            'seconds_p95':float(np.percentile(times,95)) if times else None,
            'usage_totals':dict(totals),
            'response_models':dict(Counter(str(a.get('response_model')) for a in answered))}
        for kind,total in [('action',16),('ask',6)]:
            part=[r for r in group if r['kind']==kind]
            summary['models'][model][kind]={'planned':total,'answered':len(part),
                'strict_exact':sum(r['scores']['strict_exact'] for r in part),
                'semantic_correct':sum(r['scores']['semantic_correct'] for r in part),
                'ask_type_correct':sum(r['scores']['ask_type_correct'] is True for r in part),
                'ask_focus_correct':sum(r['scores']['ask_focus_correct'] is True for r in part)}
    by_key={(r['case_id'],r['model']):r for r in rows}
    pairs=[(by_key[(c['id'],MODELS[0])],by_key[(c['id'],MODELS[1])]) for c in manifest['cases']
           if all((c['id'],model) in by_key for model in MODELS)]
    summary['paired']={'complete_pairs':len(pairs),
        'reasoner_only_semantic_correct':sum(a['scores']['semantic_correct'] and not b['scores']['semantic_correct'] for a,b in pairs),
        'chat_only_semantic_correct':sum(b['scores']['semantic_correct'] and not a['scores']['semantic_correct'] for a,b in pairs)}
    book['scores']=rows
    book['summary']=summary
    return summary


def retry_delay(response, epoch):
    value=response.headers.get('Retry-After','')
    try:
        return max(0,float(value))
    except ValueError:
        try:
            return max(0,parsedate_to_datetime(value).timestamp()-epoch)
        except (ValueError,TypeError,OverflowError):
            return 30.0


def attempt_allowed(book):
    return book['manifest']['prior_campaign_attempts']+len(book['attempts']) < book['manifest']['max_cumulative_attempts']


def run(out, manifest, *, resume=False, score_only=False):
    out.parent.mkdir(parents=True,exist_ok=True)
    with exclusive(out.with_suffix('.lock')):
        if out.exists():
            if not resume and not score_only:
                raise ValueError('Existing evidence retained; use --resume or a new output.')
            book=json.loads(out.read_bytes())
            if not score_only and book['manifest']!=manifest:
                raise ValueError('Frozen manifest/configuration differs; do not mix runs.')
        else:
            if resume or score_only:
                raise ValueError('Cannot resume or score a missing report.')
            book={'created_utc':now(),'manifest':manifest,'state':'prepared','attempts':[]}
        if score_only:
            summarize(book)
            save(out,book)
            return book
        for attempt in book['attempts']:
            if attempt['state']=='started':
                attempt['state']='interrupted_unknown'
        def persist():
            summarize(book)
            save(out,book)
        persist()  # Inputs and gold answers are saved before key access or requests.
        key=campus_api_key()
        if not key:
            book['state']='missing_campus_key'
            persist()
            return book
        cases={c['id']:c for c in manifest['cases']}
        for job in manifest['schedule']:
            while True:
                previous=[a for a in book['attempts'] if (a['case_id'],a['model'])==(job['case_id'],job['model'])]
                if any(a['state']=='answered' for a in previous):
                    break
                is_retry=bool(previous)
                if not attempt_allowed(book):
                    book['state']='budget_exhausted'; persist(); return book
                if is_retry and sum(a.get('is_retry',False) for a in book['attempts'])>=MAX_RETRIES:
                    book['state']='retry_limit_reached'; persist(); return book
                wait=max(0,book.get('next_request_at',0)-time.time())
                if wait>60:
                    book['state']='checkpoint_backoff'; persist(); return book
                if wait:
                    time.sleep(wait)
                request_body=payload(manifest,cases[job['case_id']],job['model'])
                row={**job,'index':len(book['attempts']),'state':'started','started_utc':now(),
                     'is_retry':is_retry,'payload':request_body}
                book['attempts'].append(row)
                book['state']='running'
                book['next_request_at']=time.time()+manifest['minimum_interval_seconds']
                persist()  # Charge failed/in-flight requests before they leave the process.
                start=time.perf_counter()
                response=None
                try:
                    with httpx.Client(timeout=manifest['request_timeout_seconds'],follow_redirects=False) as client:
                        response=client.post(DEEPSEEK_ENDPOINT,headers={'Authorization':'Bearer '+key},json=request_body)
                    row['http_status']=response.status_code
                    row['state']='answered' if response.status_code==200 else 'http_error'
                    if response.status_code==200:
                        try:
                            data=response.json()
                            choice=data['choices'][0]
                            message=choice['message']
                            row.update(raw=message.get('content'),finish_reason=choice.get('finish_reason'),
                                       response_model=data.get('model'),usage=data.get('usage'))
                            reasoning=message.get('reasoning_content') or ''
                            if not isinstance(reasoning,str):
                                reasoning=json.dumps(reasoning,ensure_ascii=False)
                            row.update(reasoning_characters=len(reasoning),reasoning_sha256=digest(reasoning.encode()))
                        except (ValueError,KeyError,IndexError,TypeError,AttributeError):
                            row['parse_error']='malformed_response'
                except httpx.HTTPError as exc:
                    row.update(state='network_error',http_status=None,error_type=type(exc).__name__)
                finally:
                    row['seconds']=round(time.perf_counter()-start,3)
                    persist()
                print(json.dumps({'attempt':len(book['attempts']),**job,'state':row['state'],
                                  'http_status':row.get('http_status'),'seconds':row['seconds'],
                                  'finish_reason':row.get('finish_reason')},ensure_ascii=False),flush=True)
                if row['state']=='answered':
                    break
                transient=row['state']=='network_error' or row.get('http_status')==429 or (row.get('http_status') or 0)>=500
                if not transient:
                    book['state']='stopped_http_error'; persist(); return book
                delay=retry_delay(response,time.time()) if response is not None else 30
                book['next_request_at']=max(book['next_request_at'],time.time()+delay)
                persist()
        book['state']='complete'
        persist()
        return book


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,default=DEFAULT_OUT)
    cli.add_argument('--live',action='store_true',help='Run this explicitly authorized batch; default only prints the frozen plan.')
    cli.add_argument('--dry-run',action='store_true')
    cli.add_argument('--resume',action='store_true')
    cli.add_argument('--score-only',action='store_true')
    cli.add_argument('--max-attempts',type=int,default=48,help='Cumulative cap, including --prior-attempts.')
    cli.add_argument('--prior-attempts',type=int,default=0,help='Already charged calls from a separately saved probe.')
    cli.add_argument('--reasoner-max-tokens',type=int,default=4096)
    cli.add_argument('--interval',type=float,default=6.2)
    cli.add_argument('--timeout',type=float,default=120)
    args=cli.parse_args()
    manifest=freeze(args.reasoner_max_tokens,args.prior_attempts,args.max_attempts,args.interval,args.timeout)
    if args.dry_run or (not args.live and not args.score_only):
        print(json.dumps({'models':manifest['model_tokens'],'case_ids':[c['id'] for c in manifest['cases']],
                          'planned_requests':len(manifest['schedule']),
                          'prior_attempts':manifest['prior_campaign_attempts'],
                          'scope':manifest['scope']},ensure_ascii=False,indent=2))
    else:
        result=run(args.out,manifest,resume=args.resume,score_only=args.score_only)
        print(json.dumps({'state':result['state'],'summary':result['summary']},ensure_ascii=False,indent=2))
        if result['state']!='complete' and not args.score_only:
            raise SystemExit(2)
