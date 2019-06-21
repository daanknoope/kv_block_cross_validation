from unittest import TestCase
import kv_block_cross_validation

class TestCreate_train_test_sets_hvblock_cv(TestCase):

    def test_create_train_test_ranges_types(self):
        training, test = kv_block_cross_validation.create_train_test_ranges(100, 5, 10)

        self.assertIsInstance(training, list)
        self.assertIsInstance(test, list)
        self.assertIsInstance(training[0], list)
        self.assertIsInstance(test[0], list)
        self.assertIsInstance(training[0][0], int)
        self.assertIsInstance(test[0][0], int)

    def test_create_train_test_ranges(self):
        training, test = kv_block_cross_validation.create_train_test_ranges(10, 1, 4)
        self.assertEqual(training, [[5, 6, 7, 8, 9], [0, 1, 2, 9]])
        self.assertEqual(test, [[0, 1, 2, 3], [4, 5, 6, 7]])
