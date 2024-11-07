import unittest
from multiscale import haversine


class TestHaversine(unittest.TestCase):
    def test_same(self):
        self.assertEqual(haversine([10, 10], [10, 10]), 0)

    def test_different(self):
        self.assertEqual(round(haversine([10, 10], [20, 20]), 1), 1544.8)

    def test_negative(self):
        self.assertEqual(round(haversine([-20, 10], [20, 20]), 1), 4579.2)


if __name__ == "__main__":
    unittest.main()
