import unittest
from stats import moving_average, clamp


class T(unittest.TestCase):
    def test_ma(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_ma_full(self):
        self.assertEqual(moving_average([2, 4, 6], 3), [4.0])

    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 3), 3)


if __name__ == "__main__":
    unittest.main()
