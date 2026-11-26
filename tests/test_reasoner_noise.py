"""Offline checks for evidence provenance and unrecoverable speech-intent errors."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from scripts import evaluate_reasoner_noise as noise
from scripts.evaluate_reasoner import payload

REAL_CLIENT=httpx.Client


class NoiseDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.manifest=noise.freeze(prior_attempts=45)
        self.cases={c['id']:c for c in self.manifest['cases']}

    def attempt(self,body):
        return {'state':'answered','raw':json.dumps(body,ensure_ascii=False),'finish_reason':'stop',
                'http_status':200,'seconds':1.0,'model':noise.MODEL,'usage':{},'reasoning_characters':1}

    def test_groups_sources_and_original_intent_never_leak(self):
        self.assertEqual(len(self.manifest['cases']),16)
        self.assertEqual(sum(c['source_kind']=='synthetic_text' for c in self.cases.values()),14)
        self.assertEqual(self.cases['N09']['source_kind'],'asr_control_baseline')
        for case in self.cases.values():
            task=json.loads(payload(self.manifest,case,noise.MODEL)['messages'][1]['content'])
            self.assertEqual(task['command'],case['command'])
            self.assertFalse({'original_text','original_goal','literal_transcript_goal','expected','group','ask_focus'} & task.keys())

    def test_valid_literal_can_be_correct_while_original_intent_is_at_risk(self):
        with patch('httpx.Client.post',side_effect=AssertionError('No network in scoring')):
            for id_ in ('N13','N14','N15','N16'):
                with self.subTest(id=id_):
                    case=self.cases[id_]
                    row=noise.assess(case,self.attempt(case['literal_transcript_goal']))
                    self.assertTrue(row['raw_scores']['semantic_correct'])
                    self.assertFalse(row['original_goal_check']['compatible_with_original_goal'])
                    self.assertTrue(row['relative_to_original_intent_risky_proposal'])
                    self.assertTrue(row['relative_to_original_intent_risk_admitted_by_guard'])

    def test_clarification_is_safe_but_not_literal_recovery(self):
        case=self.cases['N14']
        row=noise.assess(case,self.attempt({'action':'ask','question':'请确认是否要向前移动。'}))
        self.assertFalse(row['raw_scores']['semantic_correct'])
        self.assertTrue(row['original_goal_check']['compatible_with_original_goal'])
        self.assertFalse(row['relative_to_original_intent_risky_proposal'])

    def test_group_denominators_do_not_merge_irrecoverable_into_recovery(self):
        attempts=[]
        for case in self.manifest['cases']:
            attempts.append({**self.attempt(case['expected']),'case_id':case['id']})
        book={'manifest':self.manifest,'attempts':attempts}
        summary=noise.summarize(book)
        self.assertEqual(summary['groups']['recoverable']['planned'],6)
        self.assertEqual(summary['groups']['clarify']['planned'],6)
        self.assertEqual(summary['groups']['unrecoverable_from_text']['planned'],4)
        self.assertEqual(summary['groups']['unrecoverable_from_text']['literal_or_recovery_semantic_correct'],4)
        self.assertEqual(summary['groups']['unrecoverable_from_text']['original_intent_risky_proposals'],4)
        self.assertEqual(summary['cumulative_campaign_attempts'],61)
        self.assertEqual(summary['new_asr_calls'],0)

    def test_budget_scope_and_recorded_source_integrity(self):
        for prior,maximum in ((64,17),(45,18),(45,15),(-1,17)):
            with self.subTest(prior=prior,maximum=maximum),self.assertRaises(ValueError):
                noise.freeze(prior,maximum)
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for rel in ('configs/reasoner_asr_noise.json','configs/planner100_finalist.txt',
                        'results/safety/sensevoice/flow_live.json',
                        'results/safety/prompt_trial_audio/transfer.wav',
                        'results/safety/prompt_trial_audio/clarify.wav'):
                p=root/rel; p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes((noise.ROOT/rel).read_bytes())
            (root/'results/safety/prompt_trial_audio/transfer.wav').write_bytes(b'tampered source')
            with patch.object(noise,'ROOT',root),patch.object(noise,'DATA',root/'configs/reasoner_asr_noise.json'),self.assertRaises(ValueError):
                noise.freeze()

    def test_ledger_precharge_retry_bound_and_key_exclusion(self):
        manifest=copy.deepcopy(self.manifest)
        manifest['cases']=manifest['cases'][:1]
        manifest['minimum_interval_seconds']=0
        with TemporaryDirectory() as folder:
            path=Path(folder)/'noise.json'
            def respond(request):
                saved=json.loads(path.read_bytes())
                self.assertEqual(saved['attempts'][-1]['state'],'started')
                return httpx.Response(503,headers={'Retry-After':'0'})
            with patch.object(noise,'campus_api_key',return_value='test-only-key'), \
                    patch('scripts.evaluate_reasoner_noise.httpx.Client',
                          side_effect=lambda **kw:REAL_CLIENT(transport=httpx.MockTransport(respond),**kw)), \
                    patch('scripts.evaluate_reasoner_noise.time.sleep'):
                book=noise.run(path,manifest)
            self.assertEqual(book['state'],'retry_limit_reached')
            self.assertEqual(len(book['attempts']),2)
            self.assertEqual(book['summary']['automatic_retries'],1)
            self.assertNotIn('test-only-key',path.read_text(encoding='utf8'))


if __name__=='__main__':
    unittest.main()
