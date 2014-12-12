import clustering
import preprocessing
import unittest
import pandas as pd
import numpy as np

class HousePriceTest(unittest.TestCase):
    def test_k_prototype(self):
        df = pd.DataFrame({'value': [2, 4, 10, 12, 3, 20, 30, 11, 25]})
        res, cluster_id = clustering.k_prototype(2, 10, df)

        c1 = sorted([x[0] for x in res[0].values])
        c2 = sorted([x[0] for x in res[1].values])
        if len(c1) > len(c2) : (c1, c2) = (c2, c1)

        self.assertEqual(c1, [20, 25, 30] )
        self.assertEqual(c2, [2, 3, 4, 10, 11, 12] )

        #print '[Test1] k-prototype is Passed'

    def test_modified_k_means_clustering(self):
        # [1] k-means
        df = pd.DataFrame({'value': [2, 4, 10, 12, 3, 20, 30, 11, 25]})
        res, cluster_id = clustering.modified_k_means_clustering(2, 10, df)

        c1 = sorted([x[0] for x in res[0].values])
        c2 = sorted([x[0] for x in res[1].values])
        if len(c1) > len(c2) : (c1, c2) = (c2, c1)

        self.assertEqual(c1, [20, 25, 30] )
        self.assertEqual(c2, [2, 3, 4, 10, 11, 12] )

        # [2] test pair_categorical_values_distance:
        res = clustering.pair_categorical_values_distance(np.array(['De Niro', 'De Niro', 'Stewart', 'Grant', 'Grant', 'Stewart']),
                                                          np.array(['Scorsese', 'Coppola', 'Hitchcock', 'Hitchcock', 'Koster', 'Koster']), 'De Niro', 'Stewart')
        self.assertEqual(res, 1.0)

        res = clustering.pair_categorical_values_distance(np.array(['De Niro', 'De Niro', 'Stewart', 'Grant', 'Grant', 'Stewart']),
                                                          np.array(['Crime', 'Crime', 'Thriller', 'Thriller', 'Comedy', 'Comedy']), 'De Niro', 'Stewart')
        self.assertEqual(res, 1.0)

        # [3] test numerical_attributes_significance:
        # test1
        df = pd.DataFrame([ ['A', 'C', 1.1], ['A', 'C', 2.9], ['A', 'D', 3.1], ['B', 'D', 4.9], ['B', 'C', 4.9], ['B', 'D', 3.2], ['A', 'D', 4.8] ])
        df = preprocessing.normalize_numerical_attributes(df)

        df = clustering.get_discretized_dataframe(df, 2)
        res = clustering.get_all_numerical_attributes_weight(df)
        self.assertEqual(res, {2: 0.7000000000000001})

        # test2
        df = pd.DataFrame([ [1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [5.0, 6.0, 7.0],
                            [5.0, 8.0, 7.0], [9.0, 8.0, 10.0], [9.0, 6.0, 10.0] ])

        df = preprocessing.normalize_numerical_attributes(df)

        df = clustering.get_discretized_dataframe(df, 5)
        res = clustering.get_all_numerical_attributes_weight(df)
        self.assertEqual(res, {0: 0.75, 1: 0.6666666666666666, 2: 0.8333333333333334})


if __name__ == '__main__':
    unittest.main()