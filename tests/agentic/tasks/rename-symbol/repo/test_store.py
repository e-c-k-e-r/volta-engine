import unittest
from report import total


class T(unittest.TestCase):
    def test_total(self):
        self.assertEqual(total("data.jsonl"), 7)


if __name__ == "__main__":
    unittest.main()
