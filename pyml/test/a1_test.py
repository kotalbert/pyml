import unittest
import assignment1 as a1
from sklearn.neighbors import KNeighborsClassifier

class Test_a1_test(unittest.TestCase):
    def test_answer1(self):
        cancer = a1.answer_one()
        self.assertEqual(cancer.index._start, 0)
        self.assertEqual(cancer.index._stop, 569)

        colnames =     ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension',
    'target']

        self.assertListEqual(list(cancer.columns.values), colnames)

    def test_answer2(self):
        target = a1.answer_two()
        self.assertEqual(target.size, 2)
        self.assertListEqual(list(target.values), [212,357])
        self.assertListEqual(list(target.index), ['malignant', 'benign'])

    def test_answer3(self):
        X,y = a1.answer_three()
        self.assertTupleEqual(X.shape, (569,30))
        self.assertTupleEqual(y.shape, (569,))

    def test_answer4(self):
        X_train, X_test, y_train, y_test = a1.answer_four()
        self.assertTupleEqual(X_train.shape, (426, 30))
        self.assertTupleEqual(X_test.shape, (143, 30))
        self.assertTupleEqual(y_train.shape, (426, ))
        self.assertTupleEqual(y_test.shape, (143, ))

    def test_answer5(self):
        knn = a1.answer_five()
        self.assertIsInstance(knn, KNeighborsClassifier)

    def test_answer6(self):
        self.assertFalse('not implemented')

    def test_answer7(self):
        self.assertTupleEqual(a1.answer_seven().shape, (143,))

    def test_answer8(self):
        self.assertFalse('not implemented')

if __name__ == '__main__':
    unittest.main()
