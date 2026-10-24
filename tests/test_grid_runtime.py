import unittest, tempfile, sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cxr_project.config import load_config
ROOT=Path(__file__).resolve().parents[1]
class GridRuntimeTests(unittest.TestCase):
 def config(self, **updates):
  c=yaml.safe_load((ROOT/'configs/g1_seed42.yaml').read_text()); c.update(updates)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.yaml';p.write_text(yaml.safe_dump(c));return load_config(p)
 def test_unknown_field_rejected(self):
  with self.assertRaisesRegex(ValueError,'unknown'):self.config(head_dropuot=.4)
 def test_disabled_experts_empty_allowed(self):
  c=self.config(experts_enabled=False,hard_label_indices=[]);self.assertEqual(c['hard_label_indices'],[])
 def test_zero_workers_allowed(self):self.assertEqual(self.config(num_workers=0)['num_workers'],0)
 def test_pretrained_normalization(self):
  from cxr_project.dataset import build_transform
  c=self.config(normalization="pretrained",model="vit_base_patch16_224.augreg_in21k")
  transform=build_transform(False,16,c)
  self.assertEqual(tuple(transform.transforms[-1].mean),(.5,.5,.5))
 def test_final_requires_no_validation(self):
  with self.assertRaisesRegex(ValueError,'final_training'):self.config(final_training=True)
 def test_boolean_strings_rejected(self):
  with self.assertRaisesRegex(ValueError,'boolean'):self.config(ema='false')
 def test_checkpoint_selects_raw_before_ema_activation(self):
  from cxr_project.runtime import select_inference_state
  state={'model':{'v':1},'ema':{'v':2},'ema_active':False}
  self.assertEqual(select_inference_state(state)[0]['v'],1)
 def test_checkpoint_explicit_inactive_ema_rejected(self):
  from cxr_project.runtime import select_inference_state
  with self.assertRaisesRegex(ValueError,'active'):select_inference_state({'model':{},'ema':{'v':2},'ema_active':False},'ema')
 def test_amp_skip_does_not_update_ema(self):
  import torch
  from cxr_project.runtime import optimizer_update
  model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.1)
  class SkipScaler:
   def is_enabled(self):return True
   def get_scale(self):return self.scale
   def unscale_(self,opt):pass
   def step(self,opt):pass
   def update(self):self.scale/=2
   scale=16.
  class EMA:
   updates=0
   def update(self,m):self.updates+=1
  ema=EMA();self.assertFalse(optimizer_update(model,opt,SkipScaler(),None,ema));self.assertEqual(ema.updates,0)

