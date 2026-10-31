import unittest
from unittest import mock


class FeatureTests(unittest.TestCase):
    def test_unknown_backend_is_rejected(self):
        from src.features import get_feature_backend

        with self.assertRaisesRegex(ValueError, "unsupported feature backend"):
            get_feature_backend("unknown")

    def test_resnet_backend_is_lazy(self):
        from src.features import ResNet18Backend

        backend = ResNet18Backend(device="cpu")
        self.assertEqual(backend.device, "cpu")
        self.assertFalse(backend.is_initialized)

    def test_image_preprocessing_replicates_grayscale_channels(self):
        import numpy as np
        from src.features import preprocess_image

        tensor = preprocess_image(np.zeros((10, 20), dtype=np.uint8))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(np.array_equal(tensor[0], tensor[1]))


if __name__ == "__main__":
    unittest.main()
