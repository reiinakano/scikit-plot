from __future__ import absolute_import
import unittest
from scikitplot import scikitplot
import warnings


class TestClassifierFactory(unittest.TestCase):

    def setUp(self):
        class Classifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

            def predict_proba(self):
                pass

        class PartialClassifier:
            def __init__(self):
                pass

            def fit(self):
                pass

            def predict(self):
                pass

            def score(self):
                pass

        class NotClassifier:
            def __init__(self):
                pass

        self.Classifier = Classifier
        self.PartialClassifier = PartialClassifier
        self.NotClassifier = NotClassifier

    def test_instance_validation(self):

        clf = self.Classifier()
        scikitplot.classifier_factory(clf)

        not_clf = self.NotClassifier()
        self.assertRaises(ValueError, scikitplot.classifier_factory, not_clf)

        partial_clf = self.PartialClassifier()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            scikitplot.classifier_factory(partial_clf)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert " not in clf. Some plots may not be possible to generate." in str(w[-1].message)

    def test_plot_learning_curve_insertion(self):

        clf = self.Classifier()
        scikitplot.classifier_factory(clf)
        assert hasattr(clf, 'plot_learning_curve')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            scikitplot.classifier_factory(clf)
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert '"plot_learning_curve" method already in clf. ' \
                   'Overriding anyway. This may result in unintended behavior.' in str(w[-1].message)

if __name__ == '__main__':
    unittest.main()
