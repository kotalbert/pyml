import unittest
import numpy as np
import assignment2 as a2

class Test_a2_test(unittest.TestCase):
    def test_answer1(self):
        ans = a2.answer_one()
        self.assertEqual(ans.shape, (4,100))

    def test_answer2(self):
        ans = a2.answer_two()

        # test if 2 dimensional touple
        self.assertEqual(len(ans),2)

        
        # test if both dimensions are arrays
        t0 = ans[0]
        t1 = ans[1]
        self.assertIsInstance(t0, np.ndarray)
        self.assertIsInstance(t1, np.ndarray)

if __name__ == '__main__':
    unittest.main()
