"""No-network checks for the paired diagnostic ledger and unmodified scoring."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx

from scripts import evaluate_reasoner as benchmark

REAL_CLIENT = httpx.Client


def fixture_manifest(count=2):
    data=benchmark.freeze(prior_attempts=1)
    data['schedule']=data['schedule'][:count]
    data['minimum_interval_seconds']=0
    return data


def completion(case, *, finish='stop'):
    return httpx.Response(200,json={'model':'test-reasoner','choices':[{
        'finish_reason':finish,'message':{'content':json.dumps(case['expected'],ensure_ascii=False),
                                        'reasoning_content':'test reasoning, do not archive verbatim'}}],
        'usage':{'prompt_tokens':100,'completion_tokens':80,'total_tokens':180}})


class ReasonerBenchmarkTests(unittest.TestCase):
    def test_frozen_composition_pairing_and_no_gold_leakage(self):
        plan=benchmark.freeze(prior_attempts=1)
        self.assertEqual((len(plan['cases']),len(plan['schedule'])),(22,44))
        self.assertEqual(sum('steps' in c['expected'] for c in plan['cases']),16)
        self.assertEqual(plan['schedule'][:4],[
            {'case_id':'P003','model':'deepseek-reasoner'},
            {'case_id':'P003','model':'deepseek-chat'},
            {'case_id':'P008','model':'deepseek-chat'},
            {'case_id':'P008','model':'deepseek-reasoner'}])
        for case in plan['cases']:
            body=benchmark.payload(plan,case,'deepseek-reasoner')
            task=json.loads(body['messages'][1]['content'])
            self.assertEqual(body['max_tokens'],4096)
            self.assertFalse({'expected','ask_focus','family','split'} & task.keys())
            self.assertEqual(task['command'],case['command'])

    def test_cumulative_budget_counts_failures_and_probe(self):
        book={'manifest':benchmark.freeze(prior_attempts=1),'attempts':[{'state':'network_error'}]*46}
        self.assertTrue(benchmark.attempt_allowed(book))
        book['attempts'].append({'state':'started'})
        self.assertFalse(benchmark.attempt_allowed(book))
        for options in ({'prior_attempts':5},{'max_attempts':49},{'interval':1},{'reasoner_tokens':160}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                benchmark.freeze(**options)

    def run_mocked(self, folder, manifest, handler, **kwargs):
        path=Path(folder)/'paired.json'
        with patch.object(benchmark,'campus_api_key',return_value='test-key'), \
                patch('scripts.evaluate_reasoner.httpx.Client',
                      side_effect=lambda **opts:REAL_CLIENT(transport=httpx.MockTransport(handler),**opts)), \
                patch('scripts.evaluate_reasoner.time.sleep'):
            return benchmark.run(path,manifest,**kwargs)

    def test_precharge_then_stop_on_auth_error_and_no_key_persisted(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'paired.json'
            def handler(request):
                snapshot=json.loads(path.read_bytes())
                self.assertEqual(snapshot['attempts'][0]['state'],'started')
                self.assertEqual(snapshot['summary']['cumulative_attempts'],2)
                return httpx.Response(401)
            report=self.run_mocked(folder,fixture_manifest(),handler)
            self.assertEqual(report['state'],'stopped_http_error')
            self.assertEqual(len(report['attempts']),1)
            self.assertNotIn('test-key',path.read_text(encoding='utf8'))

    def test_completed_response_never_retried_for_quality_and_reasoning_is_hashed(self):
        with TemporaryDirectory() as folder:
            manifest=fixture_manifest()
            case=manifest['cases'][0]
            report=self.run_mocked(folder,manifest,lambda req:completion(case,finish='length'))
            self.assertEqual((report['state'],len(report['attempts'])),('complete',2))
            for row in report['attempts']:
                self.assertGreater(row['reasoning_characters'],0)
                self.assertNotIn('reasoning_content',row)
                self.assertEqual(len(row['reasoning_sha256']),64)
            for row in report['scores']:
                self.assertFalse(row['scores']['semantic_correct'])
                self.assertEqual(row['guard']['outcome'],'unavailable')
            resumed=self.run_mocked(folder,manifest,lambda req:self.fail('duplicate request'),resume=True)
            self.assertEqual(len(resumed['attempts']),2)
            self.assertNotIn('test reasoning', (Path(folder)/'paired.json').read_text(encoding='utf8'))

    def test_rate_limit_long_wait_checkpoints_and_never_follows_redirect(self):
        for status,headers,expected in [(429,{'Retry-After':'3600'},'checkpoint_backoff'),
                                        (307,{'Location':'https://example.invalid'},'stopped_http_error')]:
            with self.subTest(status=status),TemporaryDirectory() as folder:
                report=self.run_mocked(folder,fixture_manifest(),lambda req:httpx.Response(status,headers=headers))
                self.assertEqual(report['state'],expected)
                self.assertEqual(len(report['attempts']),1)

    def test_retry_limit_is_global_and_failed_attempts_not_scored_as_answers(self):
        with TemporaryDirectory() as folder:
            report=self.run_mocked(folder,fixture_manifest(),lambda req:httpx.Response(503,headers={'Retry-After':'0'}))
            self.assertEqual(report['state'],'retry_limit_reached')
            self.assertEqual(len(report['attempts']),3)
            self.assertEqual(report['summary']['automatic_retries'],2)
            self.assertEqual(report['summary']['this_report_attempts'],3)
            self.assertEqual(report['scores'],[])

    def test_manifest_change_and_result_overwrite_are_rejected(self):
        with TemporaryDirectory() as folder:
            manifest=fixture_manifest(1)
            self.run_mocked(folder,manifest,lambda req:completion(manifest['cases'][0]))
            with self.assertRaises(ValueError):
                self.run_mocked(folder,manifest,lambda req:self.fail('unexpected'))
            changed=copy.deepcopy(manifest)
            changed['model_tokens']['deepseek-reasoner']=2048
            with self.assertRaises(ValueError):
                self.run_mocked(folder,changed,lambda req:self.fail('unexpected'),resume=True)

    def test_guard_does_not_turn_schema_success_into_semantic_success(self):
        manifest=benchmark.freeze()
        case=next(c for c in manifest['cases'] if c['id']=='P055')
        wrong={'steps':[{'action':'move','axis':'y','mm':25}]}
        row={'http_status':200,'finish_reason':'stop','raw':json.dumps(wrong)}
        scored=benchmark.score_response(case,row['raw'])
        self.assertTrue(scored['schema_valid'])
        self.assertFalse(scored['semantic_correct'])
        self.assertTrue(scored['dangerous_motion'])
        with patch('httpx.Client.post',side_effect=AssertionError('no network in guard')):
            guard=benchmark.guard_result(case,row)
        self.assertFalse(guard['wrong_motion_admitted'])
        self.assertEqual(guard['outcome'],'rejected')


if __name__=='__main__':
    unittest.main()
