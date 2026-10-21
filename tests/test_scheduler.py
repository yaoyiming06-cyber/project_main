import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_project.schedule import scheduler_step_counts


class SchedulerStepCountsTests(unittest.TestCase):
    def test_counts_optimizer_updates_after_gradient_accumulation(self):
        self.assertEqual(
            scheduler_step_counts(9257, 8, epochs=10, warmup_epochs=5),
            (11580, 5790),
        )

    def test_counts_short_final_accumulation_window(self):
        self.assertEqual(
            scheduler_step_counts(9, 4, epochs=3, warmup_epochs=1),
            (9, 3),
        )

    def test_rejects_invalid_schedule_dimensions(self):
        for arguments in ((0, 4, 2, 1), (4, 0, 2, 1), (4, 2, 0, 0), (4, 2, 2, 3)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                scheduler_step_counts(*arguments)


if __name__ == "__main__":
    unittest.main()
