import unittest
from textutil import title, truncate


class T(unittest.TestCase):
    def test_title(self):
        self.assertEqual(title("hello big world"), "Hello Big World")

    def test_truncate(self):
        self.assertEqual(truncate("abcdef", 4), "abc…")


if __name__ == "__main__":
    unittest.main()
