import unittest
import os
from ..utils import *


class TestNearestNeighbor(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(TestNearestNeighbor, self).__init__(methodName)
        self._test_package_path = __file__.split(os.sep)[:-1]
        self._resource_path = os.sep.join(self._test_package_path) + os.sep + "test_resources" + os.sep
        self._fix = self._resource_path + "fix.json"

    def test_identical(self):
        """ All keypoints are identical.
        """
        store = self._resource_path + "0_w.json"
        self.assertEqual(nearest_neighbor(store, self._fix), ({0: 1.0}, {0: 0}))

    def test_dropout(self):
        """ Keypoint index 0 and 2 are not present anymore. No misclassifications should appear.
        """
        store = self._resource_path + "1_w.json"
        self.assertEqual(nearest_neighbor(store, self._fix), ({1: 1.0}, {1: 0}))

    def test_misclassified(self):
        """ Keypoint index 0 and 2 are swapped and 1 and 3. A total of 4 misclassifications should appear.
        """
        store = self._resource_path + "2_w.json"
        self.assertEqual(nearest_neighbor(store, self._fix), ({2: 0.9876543209876543}, {2: 4}))


def run():
    unittest.main(module='endolas.test.test_utils')
