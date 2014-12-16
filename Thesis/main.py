# coding=utf-8
import HousePricePrediction.clustering as clustering
import HousePricePrediction.preprocessing as preprocessing
import pandas as pd
import time
import math
import ConfigParser

def main():

    start_time = time.time()

    # read some configuration from Config.ini
    config = ConfigParser.ConfigParser()
    config.read('Config.ini')
    Clustering_Algorithm = config.get('Clustering', 'Clustering_Algorithm')
    N_Run = int(config.get('Clustering', 'N_Run'))
    N_Clusters = int(config.get('Clustering', 'N_Clusters'))
    Max_Iters = int(config.get('Clustering', 'Max_Iters'))

    # read data into dataframe
    df = pd.read_csv(config.get('Config', 'Data_Path'))

    # in order to get clustering results in original dataframe
    original_df = df.copy()

    # data preprocessing
    df = preprocessing.feature_selection(df)
    df = preprocessing.fill_NaN(df)
    df = preprocessing.normalize_numerical_attributes(df)

    cnt = 0
    metric = {'WSS': 0.0, 'BSS': 0.0, 'CV': 0.0}
    while cnt < N_Run:
        # clustering
        if Clustering_Algorithm == 'k_prototype':
            print '-> Start k_prototype Algorithm'
            normalized_cluster_df_list, cluster_id = clustering.k_prototype(N_Clusters, Max_Iters, df)
        elif Clustering_Algorithm == 'modified_k_means_clustering':
            print 'Start modified_k_means_clustering Algorithm'
            normalized_cluster_df_list, cluster_id = clustering.modified_k_means_clustering(N_Clusters, Max_Iters, df) #

        print '-> Clustering Results: '
        for key, value in cluster_id.iteritems():
            print '# Cluster ' + str(key + 1) + ': ',
            print str(value) + '\n'

        # get original dataframe by using cluster_id
        cluster_df_list = [ original_df.iloc[cluster_id[c]] for c in xrange(0, len(cluster_id)) ]

        # evaluation clustering
        wss, bss, cv = clustering.evaluate_clustering(cluster_df_list, normalized_cluster_df_list, Clustering_Algorithm)

        # prevent from empty cluster
        if math.isnan(wss) or math.isnan(bss) or math.isnan(cv):
            continue

        metric['WSS'] += wss
        metric['BSS'] += bss
        metric['CV'] += cv
        cnt += 1

    metric['WSS'] /= float(N_Run)
    metric['BSS'] /= float(N_Run)
    metric['CV'] /= float(N_Run)

    print 'WSS: ' + str(metric['WSS'])
    print 'BSS: ' + str(metric['BSS'])
    print 'CV: ' + str(metric['CV']) + '%'

    print '-> Total Time: ' + str(time.time() - start_time)
if __name__ == '__main__':
    main()
