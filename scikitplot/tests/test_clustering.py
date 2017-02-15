from __future__ import absolute_import
import unittest
import scikitplot
import warnings


class TestClassifierFactory(unittest.TestCase):

    def setUp(self):
        class Clusterer:
            def __init__(self):
                pass

            def fit(self):
                pass

            def fit_predict(self):
                pass

        class NotClusterer:
            def __init__(self):
                pass

        self.Clusterer = Clusterer
        self.NotClusterer = NotClusterer

    def test_instance_validation(self):

        clf = self.Clusterer()
        scikitplot.clustering_factory(clf)

        not_clf = self.NotClusterer()
        self.assertRaises(TypeError, scikitplot.clustering_factory, not_clf)

    def test_method_insertion(self):

        clf = self.Clusterer()
        scikitplot.clustering_factory(clf)
        assert hasattr(clf, 'plot_silhouette')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            scikitplot.clustering_factory(clf)

            assert len(w) == 1
            for warning in w:
                assert issubclass(warning.category, UserWarning)
                assert ' method already in clf. ' \
                       'Overriding anyway. This may ' \
                       'result in unintended behavior.' in str(warning.message)

if __name__ == '__main__':
    unittest.main()
