import importlib.util
import sys
import unittest
import tempfile
import csv
import json
from unittest.mock import patch
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('predict_test',ROOT/'scripts/predict_test.py')
predict=importlib.util.module_from_spec(spec);spec.loader.exec_module(predict)

class ToyModel(nn.Module):
    hard_label_indices=()
    def forward(self, images, offsets, views):
        logits=torch.stack([images[a:b,...,0].mean() for a,b in zip(offsets[:-1],offsets[1:])])
        return logits[:,None].expand(-1,10),logits.new_empty((len(logits),0))

class InferenceTests(unittest.TestCase):
    def test_flip_averages_probabilities_not_logits(self):
        images=torch.tensor([[[[0.,2.],[0.,2.]]]])
        batch={'images':images,'offsets':[0,1],'view_ids':torch.tensor([0])}
        result=predict.predict_batch(ToyModel(),batch,torch.device('cpu'),False,True)
        torch.testing.assert_close(result,torch.full((1,10),(.5+torch.sigmoid(torch.tensor(2.)).item())/2))

    def test_config_rejects_wrong_resolution_and_architecture(self):
        with self.assertRaisesRegex(ValueError,'input_size'):
            predict.validate_checkpoint_config({'model':'x','input_size':224},{'config':{'model':'x','input_size':256}})
        with self.assertRaisesRegex(ValueError,'model'):
            predict.validate_checkpoint_config({'model':'x','input_size':224},{'config':{'model':'y','input_size':224}})

    def test_swin_scale_switching_restores_original_weights(self):
        from timm.models.swin_transformer import SwinTransformer
        from cxr_project.model import ConvNeXtStudyClassifier
        backbone=SwinTransformer(img_size=192,embed_dim=8,depths=(1,1,1,1),
                                  num_heads=(1,2,4,8),window_size=7,num_classes=0)
        model=ConvNeXtStudyClassifier(backbone=backbone,feature_dim=64,pretrained=False).eval()
        source={key:value.clone() for key,value in model.state_dict().items()}
        config={'model':'swin_base_patch4_window7_224.ms_in22k','input_size':192}
        predict.set_inference_size(model,config,224,source)
        first={key:value.clone() for key,value in model.state_dict().items()}
        predict.set_inference_size(model,config,256,source)
        predict.set_inference_size(model,config,224,source)
        for key,value in first.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)

    def test_inactive_ema_uses_raw_and_casts_half(self):
        state,source=predict.inference_weights({'model':{'w':torch.ones(1).half()},'ema':{'w':torch.zeros(1)},'ema_active':False})
        self.assertEqual(source,'raw');self.assertEqual(state['w'].dtype,torch.float32)
        self.assertEqual(state['w'].item(),1.)

    def test_compact_ema_export_can_be_selected_by_origin(self):
        state,source=predict.inference_weights({'model':{'w':torch.ones(1).half()},'weight_source':'ema','format':'cxr_fp16_inference_v1','ema_active':False},'ema')
        self.assertEqual(source,'ema')
        with self.assertRaisesRegex(ValueError,'raw'):
            predict.inference_weights({'model':{'w':torch.ones(1)},'weight_source':'ema','format':'cxr_fp16_inference_v1'},'raw')

    def test_labeled_multiscale_cli_exports_view_coverage(self):
        from PIL import Image
        from cxr_project.model import ConvNeXtStudyClassifier
        def tiny(*args,**kwargs):
            backbone=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,4))
            return ConvNeXtStudyClassifier(backbone=backbone,feature_dim=4,pretrained=False,
                                          experts_enabled=False,gradient_checkpointing=False)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=root/'valid.csv';output=root/'prediction.csv';checkpoint=root/'checkpoint.pth'
            config={'model':'convnext_base.fb_in22k','input_size':32,'num_workers':0,
                    'batch_size_studies':1,'pin_memory':False,'amp':False,'experts_enabled':False,
                    'normalization':'imagenet1k','view_filter':['AP'],'augmentation':'G0'}
            with manifest.open('w',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['image_path','Subject_id','Study_id','Dicom_id','ViewPosition',*[f'label_{i}' for i in range(10)]])
                for i,view in enumerate(['AP','PA']):
                    Image.new('RGB',(40,24),(60+i*80,80,100)).save(root/f'{i}.png')
                    writer.writerow([f'{i}.png',f'subject_{i}',f'study_{i}',str(i),view,*([i]*10)])
            torch.save({'model':tiny().state_dict(),'config':config,'ema_active':False},checkpoint)
            argv=['predict_test.py','--config',str(root/'config.yaml'),'--checkpoint',str(checkpoint),
                  '--data-root',str(root),'--output',str(output),'--manifest',str(manifest),
                  '--labeled','--tta','horizontal-flip','--tta-sizes','32','48','--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(predict,'load_config',return_value=config),patch.object(predict,'build_model',side_effect=tiny):
                predict.main()
            with output.open() as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['Study_id'],'study_0')
            self.assertIn('target_0',rows[0]);self.assertIn('probability_9',rows[0])
            metadata=json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(metadata['coverage']['dropped_study_ids'],['study_1'])
            self.assertEqual(metadata['tta_sizes'],[32,48]);self.assertEqual(metadata['weight_source'],'raw')

if __name__=='__main__':unittest.main()
