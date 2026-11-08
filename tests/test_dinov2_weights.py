import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cxr_project.general_weights import weight_source


class DINOv2WeightTests(unittest.TestCase):
    def setUp(self):
        from cxr_project import dinov2_weights

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.path = self.root / "dinov2_vitb14_pretrain.pth"
        self.payload = b"synthetic DINOv2 state dict checkpoint"
        self.path.write_bytes(self.payload)
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.source = {
            "model": "vit_base_patch14_dinov2",
            "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            "bytes": len(self.payload),
            "sha256": self.digest,
            "path": str(self.path.resolve()),
        }
        self.write_source()
        self.pinned_weights_patch = patch.multiple(
            dinov2_weights,
            WEIGHTS_BYTES=len(self.payload),
            WEIGHTS_SHA256=self.digest,
        )
        self.pinned_weights_patch.start()

    def tearDown(self):
        self.pinned_weights_patch.stop()
        self.temp_dir.cleanup()

    def write_source(self, **changes):
        record = {**self.source, **changes}
        self.path.with_suffix(self.path.suffix + ".source.json").write_text(
            json.dumps(record), encoding="utf-8",
        )

    def test_official_general_source_is_allowlisted(self):
        source = weight_source("vit_base_patch14_dinov2")
        self.assertEqual(source["url"], self.source["url"])
        self.assertEqual(source["filename"], self.path.name)
        self.assertIn("LVD-142M", source["dataset"])
        with self.assertRaises(ValueError):
            weight_source("xray_dino")

    def test_valid_local_file_and_provenance_record_pass(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        result = validate_checkpoint(self.path, self.digest)
        self.assertEqual(result["sha256"], self.digest)
        self.assertEqual(result["bytes"], len(self.payload))

    def test_missing_provenance_record_is_rejected(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        self.path.with_suffix(self.path.suffix + ".source.json").unlink()
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_checkpoint(self.path, self.digest)

    def test_wrong_url_is_rejected(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        self.write_source(url="https://example.invalid/not-dinov2.pth")
        with self.assertRaisesRegex(ValueError, "official source"):
            validate_checkpoint(self.path, self.digest)

    def test_changed_file_size_is_rejected(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        with self.path.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "size"):
            validate_checkpoint(self.path, self.digest)

    def test_sidecar_digest_mismatch_is_rejected(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        self.write_source(sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_checkpoint(self.path, self.digest)

    def test_configured_digest_mismatch_is_rejected(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        with self.assertRaisesRegex(ValueError, "configured.*SHA-256"):
            validate_checkpoint(self.path, "0" * 64)

    def test_invalid_configured_digest_is_rejected(self):
        from cxr_project.dinov2_weights import validate_checkpoint

        with self.assertRaisesRegex(ValueError, "64-character SHA-256"):
            validate_checkpoint(self.path, "bad-digest")

    def test_downloader_writes_checkpoint_and_provenance_atomically(self):
        from cxr_project import dinov2_weights

        output = self.root / "downloaded.pth"
        payload = b"download response bytes"
        with patch("cxr_project.dinov2_weights.urllib.request.urlopen", return_value=io.BytesIO(payload)), \
             patch.object(dinov2_weights, "WEIGHTS_BYTES", len(payload)), \
             patch.object(dinov2_weights, "WEIGHTS_SHA256", hashlib.sha256(payload).hexdigest()):
            record = dinov2_weights.download_checkpoint(output)
        self.assertEqual(output.read_bytes(), payload)
        self.assertEqual(record["bytes"], len(payload))
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
        sidecar = json.loads(output.with_suffix(".pth.source.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["url"], self.source["url"])
        self.assertFalse(output.with_suffix(".pth.part").exists())
        self.assertFalse(output.with_suffix(".pth.source.json.part").exists())


if __name__ == "__main__":
    unittest.main()
