import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import yaml
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr_project.model import ConvNeXtStudyClassifier
from cxr_project.train import train
ROOT=Path(__file__).resolve().parents[1]

def tiny(config,**kwargs):
    model=ConvNeXtStudyClassifier(backbone=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten()),feature_dim=3,pretrained=False,hard_label_indices=[],gradient_checkpointing=False,image_chunk_size=10)
    return model

class TrainGridTest(unittest.TestCase):
    def fixture(self,root,final=False):
        cfg=yaml.safe_load((ROOT/'configs/g1_seed42.yaml').read_text())
        cfg.update(project_root=str(root),pretrained=False,epochs=1,warmup_epochs=0,num_workers=0,batch_size_studies=2,accum_steps=1,input_size=16,gradient_checkpointing=False,amp=False,train_manifest='data.csv',valid_manifest=None if final else 'data.csv',image_root='.',output_dir='run',ema=True,ema_start_epoch=1,evaluate_raw_and_ema=not final,experts_enabled=False,hard_label_indices=[],pos_weight_mode='unit',sampling_strategy='weighted',mixup_alpha=.2,loss='asl',final_training=final,early_stopping_patience=0,augmentation='G0')
        with (root/'data.csv').open('w',newline='') as handle:
            columns=['image_path','Subject_id','Study_id','ViewPosition']+[f'label_{i}' for i in range(10)]
            w=csv.DictWriter(handle,fieldnames=columns);w.writeheader()
            for i in range(4):
                for view in range(1+i%2):
                    name=f'{i}_{view}.png';Image.new('RGB',(16,16),color=(30+i*50,)*3).save(root/name)
                    w.writerow(dict(image_path=name,Subject_id=str(i),Study_id=str(i),ViewPosition='AP',**{f'label_{j}':i%2 for j in range(10)}))
        path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg));return path

    def test_raw_ema_mixup_sampler_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=self.fixture(root)
            with patch('cxr_project.train.build_model',side_effect=tiny):train(cfg,device_name='cpu')
            for source in ('raw','ema'):
                checkpoint=torch.load(root/'run'/f'checkpoint-best-{source}.pth',weights_only=False)
                self.assertEqual(checkpoint['selected_weights'],source)
                self.assertIn('rng_state',checkpoint)
                self.assertTrue((root/'run'/f'valid_best_{source}_predictions.csv').is_file())

    def test_final_mode_never_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=self.fixture(root,True)
            with patch('cxr_project.train.build_model',side_effect=tiny),patch('cxr_project.train.evaluate',side_effect=AssertionError('must not validate')):train(cfg,device_name='cpu')
            self.assertTrue((root/'run'/'checkpoint-final.pth').is_file())
            self.assertFalse((root/'run'/'checkpoint-best.pth').exists())

    def test_epoch_resume_matches_uninterrupted_training(self):
        import cxr_project.train as module
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.fixture(root)
            cfg=yaml.safe_load(path.read_text());cfg['epochs']=2;path.write_text(yaml.safe_dump(cfg))
            with patch('cxr_project.train.build_model',side_effect=tiny):
                train(path,output_dir=root/'continuous',device_name='cpu')
            original=module._train_epoch
            def interrupted(*args,**kwargs):
                if args[7]==1: raise KeyboardInterrupt('simulated interruption')
                return original(*args,**kwargs)
            with patch('cxr_project.train.build_model',side_effect=tiny),patch('cxr_project.train._train_epoch',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt): train(path,output_dir=root/'interrupted',device_name='cpu')
            with patch('cxr_project.train.build_model',side_effect=tiny):
                train(path,output_dir=root/'interrupted',resume=root/'interrupted'/'checkpoint-last.pth',device_name='cpu')
            a=torch.load(root/'continuous'/'checkpoint-last.pth',weights_only=False)
            b=torch.load(root/'interrupted'/'checkpoint-last.pth',weights_only=False)
            for key,value in a['model'].items():torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
            self.assertEqual(a['global_step'],b['global_step'])

if __name__=='__main__':unittest.main()
