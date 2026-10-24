import csv
import tempfile
import unittest
from pathlib import Path

from cxr_project.grid import experiment_overrides, build_configs
from cxr_project.evaluation_tools import (read_predictions, aligned_predictions, make_subject_folds, merge_final_manifests, fuse_predictions, paired_bootstrap)


class GridToolsTests(unittest.TestCase):
    def test_single_factor_and_seed_paths(self):
        base = {'seed': 42, 'run_name': 'old', 'output_dir': 'runs/old', 'input_size': 224, 'batch_size_studies': 2, 'accum_steps': 8, 'train_manifest': 'data/train.csv'}
        generated = build_configs(base, ['I4', 'I5', 'I6'], [42, 2027, 3407], Path('/tmp/project'), Path('/tmp/runs'))
        self.assertEqual(len(generated), 9)
        self.assertEqual(len({c['output_dir'] for c in generated.values()}), 9)
        self.assertEqual({c['input_size'] for c in generated.values()}, {192,224,256})
        self.assertTrue(all(c['train_manifest']==str(Path('/tmp/project/data/train.csv').resolve()) for c in generated.values()))
        self.assertEqual(experiment_overrides('M1')['experts_enabled'], False)
        with self.assertRaises(ValueError): experiment_overrides('V2')
        with self.assertRaises(ValueError): build_configs(base, ['B1'], [42], Path('/tmp/project'), Path('/tmp/runs'))

    def test_export_selects_recorded_validated_raw_weights(self):
        import torch
        from cxr_project.evaluation_tools import export_fp16
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            source=p/'source.pth'
            torch.save({'model':{'w':torch.tensor([1.])},'ema':{'w':torch.tensor([2.])},'ema_active':True,'weight_source':'raw','selected_weights':'raw','config':{}},source)
            export_fp16(source,p/'export.pth')
            exported=torch.load(p/'export.pth',weights_only=False)
            self.assertEqual(exported['model']['w'].item(),1.)
            self.assertEqual(exported['model']['w'].dtype,torch.float16)
            self.assertNotIn('optimizer',exported)

    def _csv(self, path, rows):
        with path.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return path

    def test_patient_folds_and_final_overlap_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            rows=[{'Subject_id':f'p{i//2}', 'Study_id':f's{i}', 'image_path':f'{i}.jpg', **{f'label_{j}':str(i%2) for j in range(10)}} for i in range(12)]
            source=self._csv(p/'train.csv', rows)
            result=make_subject_folds(source,p/'folds',3,42)
            seen=[]
            for fold in result['folds']:
                with open(fold['train_manifest'], encoding='utf-8-sig') as handle: tr=list(csv.DictReader(handle))
                with open(fold['valid_manifest'], encoding='utf-8-sig') as handle: va=list(csv.DictReader(handle))
                self.assertFalse({r['Subject_id'] for r in tr}&{r['Subject_id'] for r in va})
                seen.extend(r['Study_id'] for r in va)
            self.assertEqual(sorted(seen),sorted(r['Study_id'] for r in rows))
            with self.assertRaises(ValueError): merge_final_manifests(source,source,p/'merged.csv',final=True)
            with self.assertRaises(ValueError): merge_final_manifests(source,source,p/'merged.csv',final=False)

    def test_alignment_targets_fusion_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            def row(i,prob): return {'Study_id':f's{i}',**{f'target_{j}':i%2 for j in range(10)},**{f'probability_{j}':prob for j in range(10)}}
            a=self._csv(p/'a.csv',[row(i,.9 if i%2 else .1) for i in range(8)])
            b=self._csv(p/'b.csv',[row(i,.8 if i%2 else .2) for i in reversed(range(8))])
            manifest=self._csv(p/'valid.csv',[{'Study_id':f's{i}','Subject_id':f'p{i//2}'} for i in range(8)])
            result=fuse_predictions([a,b],p/'fused.csv')
            self.assertAlmostEqual(result['macro_auc'],1)
            boot=paired_bootstrap(a,b,manifest,iterations=20,seed=42)
            self.assertEqual(boot['delta_ci95'],[0.,0.])
            with open(b) as handle: bad=list(csv.DictReader(handle))
            bad[0]['target_0']='0'
            self._csv(p/'bad.csv',bad)
            with self.assertRaises(ValueError): aligned_predictions([read_predictions(a),read_predictions(p/'bad.csv')])

if __name__=='__main__': unittest.main()

class RoutingTests(unittest.TestCase):
    def test_route_keeps_missing_view_and_unselected_labels(self):
        from cxr_project.evaluation_tools import write_predictions, route_predictions
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            write_predictions(p/'base.csv',['s0','s1'],None,np.full((2,10),.2))
            write_predictions(p/'expert.csv',['s0'],None,np.full((1,10),.8))
            with (p/'manifest.csv').open('w') as handle:
                handle.write('Study_id,ViewPosition\ns0,AP\ns1,PA\n')
            report=route_predictions(p/'base.csv',{'AP':p/'expert.csv'},p/'manifest.csv',p/'out.csv',label_indices=[0,3,7])
            actual=read_predictions(p/'out.csv')['probabilities']
            self.assertEqual(report['fallback_studies'],1)
            self.assertAlmostEqual(actual[0,0],.5)
            self.assertAlmostEqual(actual[0,1],.2)
            self.assertAlmostEqual(actual[1,0],.2)

class TestPredictionFormats(unittest.TestCase):
    def test_test_csv_format_and_selected_label_fusion(self):
        from cxr_project.evaluation_tools import fuse_predictions
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,prob in [('a',.2),('b',.8)]:
                with (p/(name+'.csv')).open('w',newline='') as handle:
                    writer=csv.writer(handle);writer.writerow(['Study_id',*[f'label_{j}' for j in range(10)]]);writer.writerow(['s0',*[prob]*10])
            fuse_predictions([p/'a.csv',p/'b.csv'],p/'out.csv',label_indices=[0,3,7])
            result=read_predictions(p/'out.csv')['probabilities']
            self.assertAlmostEqual(result[0,0],.5)
            self.assertAlmostEqual(result[0,1],.2)
