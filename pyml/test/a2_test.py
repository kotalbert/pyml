import unittest
import assignment2 as a2

class Test_a2_test(unittest.TestCase):
    def test_answer1(self):
        ans = a2.answer_one()
        self.assertEqual(ans.shape, (4,100))

if __name__ == '__main__':
    unittest.main()
