# tests/test_normalization.py
import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pybrain.core.normalization import norm01, zscore_robust

class TestNormalization(unittest.TestCase):
    def test_norm01(self):
        data = np.array([0, 5, 10], dtype=np.float32)
        normed = norm01(data)
        self.assertEqual(normed.min(), 0.0)
        self.assertEqual(normed.max(), 1.0)
        self.assertEqual(normed[1], 0.5)

    def test_zscore_robust(self):
        # Create data with some variance and outliers
        np.random.seed(42)
        data = np.random.normal(100, 10, (10, 10, 10))
        data[0,0,0] = 1000 # outlier
        mask = np.ones((10, 10, 10))
        
        normed = zscore_robust(data, mask)
        # Mean should be around 0, std around 1 after robust scaling
        self.assertAlmostEqual(float(normed.mean()), 0.0, places=1)
        self.assertAlmostEqual(float(normed.std()), 1.0, places=1)

if __name__ == "__main__":
    unittest.main()
