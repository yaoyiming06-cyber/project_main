import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from PIL import Image
from cxr_project.dataset import build_transform
from cxr_project.losses import compute_training_loss
from cxr_project.manifest import StudyRecord

class GridDataTests(unittest.TestCase):
    def test_g0_ignores_all_random_flags(self):
        im = Image.fromarray(np.arange(16*16*3, dtype=np.uint8).reshape(16,16,3))
        cfg = dict(augmentation='G0', horizontal_flip=True, horizontal_flip_prob=1., gamma_prob=1., gamma_range=[.5,.5], random_erasing_prob=1.)
        a = build_transform(True, 16, cfg)(im)
        b = build_transform(False, 16, cfg)(im)
        torch.testing.assert_close(a,b)

    def test_aliases_and_deterministic_clahe(self):
        im = Image.fromarray(np.tile(np.arange(16,dtype=np.uint8), (16,1))).convert('RGB')
        for strategy in ['G2','G3']:
            self.assertEqual(build_transform(True,16,dict(augmentation=strategy))(im).shape,(3,16,16))
        cfg = dict(augmentation='G0', clahe=True)
        a,b = build_transform(False,16,cfg)(im),build_transform(False,16,cfg)(im)
        torch.testing.assert_close(a,b)
        self.assertFalse(torch.equal(a,build_transform(False,16,dict(augmentation='G0'))(im)))

    def test_asl_masks_uncertain_targets(self):
        main = torch.zeros(2,10,requires_grad=True)
        expert = torch.empty(2,0)
        targets = torch.zeros(2,10); targets[:,0] = -1
        loss,_ = compute_training_loss(main,expert,targets,torch.ones(10),hard_label_indices=[],loss_name='asl')
        loss.backward()
        torch.testing.assert_close(main.grad[:,0],torch.zeros(2))
        self.assertTrue(torch.isfinite(loss))

    def test_gamma_and_erasing_affect_train_only(self):
        image = Image.new('RGB',(32,32),(80,80,80))
        base = dict(augmentation='G1',augmentation_degrees=0,augmentation_translate=0,augmentation_scale=[1,1],color_jitter_brightness=0,color_jitter_contrast=0)
        reference = build_transform(False,32,base)(image)
        for extra in [dict(gamma_prob=1.,gamma_range=[.5,.5]),dict(random_erasing_prob=1.,random_erasing_scale=[.1,.1],random_erasing_ratio=[1.,1.])]:
            cfg = dict(base,**extra)
            self.assertFalse(torch.equal(reference,build_transform(True,32,cfg)(image)))
            torch.testing.assert_close(reference,build_transform(False,32,cfg)(image))

    def test_asl_zero_gamma_and_clip_matches_bce(self):
        main = torch.randn(2,10)
        targets = torch.randint(0,2,(2,10)).float(); targets[0,0]=-1
        kwargs = dict(hard_label_indices=[])
        bce,_ = compute_training_loss(main,torch.empty(2,0),targets,torch.ones(10),**kwargs)
        asl,_ = compute_training_loss(main,torch.empty(2,0),targets,torch.ones(10),loss_name='asl',asl_gamma_neg=0,asl_gamma_pos=0,asl_clip=0,**kwargs)
        torch.testing.assert_close(bce,asl)

    def test_sampling_is_seeded_and_bounded_and_requires_unit_weights(self):
        from cxr_project.sampling import build_train_sampler, study_sampling_weights
        studies = [StudyRecord(str(i),str(i),('a',),('AP',),tuple([int(i==0)]+[0]*9)) for i in range(20)]
        weights = study_sampling_weights(studies, power=.5,max_weight=3.)
        self.assertLessEqual(float(weights.max()/weights.min()),3.)
        cfg = dict(sampling_strategy='weighted',pos_weight_mode='unit')
        self.assertEqual(list(build_train_sampler(studies,cfg,42)),list(build_train_sampler(studies,cfg,42)))
        with self.assertRaisesRegex(ValueError,'pos_weight'):
            build_train_sampler(studies,dict(sampling_strategy='weighted',pos_weight_mode='ratio'),42)

    def test_view_filter_preserves_studies_and_reports_coverage(self):
        from cxr_project.dataset import filter_study_views
        studies = [StudyRecord('a','s',('a','b','c'),('AP','AP','LL'),(0,)*10),StudyRecord('b','t',('d',),('PA',),(0,)*10)]
        filtered,coverage = filter_study_views(studies,['AP'])
        self.assertEqual(len(filtered),1)
        self.assertEqual(filtered[0].image_paths,('a','b'))
        self.assertEqual(coverage['retained_studies'],1)
        self.assertEqual(coverage['total_studies'],2)

    def test_dataset_without_view_embedding_does_not_require_view_position(self):
        import csv
        import tempfile
        from cxr_project.dataset import StudyDataset, collate_studies
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB',(8,8),'white').save(root/'one.png')
            Image.new('RGB',(8,8),'black').save(root/'two.png')
            manifest = root/'train.csv'
            columns = ['image_path','Subject_id','Study_id']+[f'label_{i}' for i in range(10)]
            with manifest.open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
                for index,name in enumerate(('one.png','two.png')):
                    writer.writerow(dict(image_path=name,Subject_id='subject',Study_id='study',**{f'label_{i}':0 for i in range(10)}))
            dataset = StudyDataset(manifest,root,transform=build_transform(False,8),
                                   view_filter=None,use_view_embedding=False)
            batch = collate_studies([dataset[0]])
            self.assertEqual(batch['view_ids'].tolist(),[0,0])
            self.assertEqual(batch['offsets'].tolist(),[0,2])
            with self.assertRaisesRegex(ValueError,'ViewPosition'):
                StudyDataset(manifest,root,view_filter=['AP'],use_view_embedding=False)

    def test_mixup_unknown_targets_not_fabricated(self):
        from cxr_project.losses import mixed_training_loss
        main = torch.zeros(2,10,requires_grad=True)
        expert = torch.empty(2,0)
        targets = torch.zeros(2,10); targets[:,0] = -1
        permutation = torch.tensor([1,0])
        loss,_ = mixed_training_loss(main,expert,targets,torch.ones(10),permutation,.3,hard_label_indices=[])
        loss.backward()
        torch.testing.assert_close(main.grad[:,0],torch.zeros(2))

    def test_normalization_tracks_filtered_clahe_images(self):
        import csv
        import tempfile
        from cxr_project.dataset import compute_dataset_normalization, CLAHE, ResizePadToSquare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.fromarray(np.tile(np.arange(16,dtype=np.uint8), (16,1))).convert('RGB')
            image.save(root/'ap.png')
            Image.new('RGB',(16,16),'white').save(root/'pa.png')
            manifest = root/'train.csv'
            with manifest.open('w',newline='') as stream:
                columns = ['image_path','Subject_id','Study_id','ViewPosition']+[f'label_{i}' for i in range(10)]
                writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
                for i,view in enumerate(['AP','PA']):
                    writer.writerow(dict(image_path=view.lower()+'.png',Subject_id=str(i),Study_id=str(i),ViewPosition=view,**{f'label_{j}':0 for j in range(10)}))
            stats = compute_dataset_normalization(manifest,root,16,config=dict(view_filter=['AP'],clahe=True))
            expected = np.asarray(ResizePadToSquare(16)(CLAHE()(image)),dtype=np.float64).mean((0,1))/255.
            np.testing.assert_allclose(stats['mean'],expected,rtol=1e-6)
            self.assertEqual(stats['images'],1)

if __name__ == '__main__': unittest.main()
