# coding=utf-8


import random
import math
import sys
import re
import __main__
import numpy as np
import pandas as pd
import preprocessing
import logging
from munkres import Munkres
from collections import defaultdict
from collections import Counter
import distance_fun


categorical_attributes_distance = {}
numerical_attributes_weight = {}
categorical_attributes_weight = []


# def kp_distance(object_x, object_y, Wc):
#     ''' Distance function for two objects
#         '''
#     res = 0.0
#     for attr_idx in xrange(0, len(object_x)):
#         cur_type = type(object_x[attr_idx])
#         if( (cur_type == str) or (cur_type == np.string_) ):
#             # categorical attribute
#             if (object_x[attr_idx] == object_y[attr_idx]):
#                 res = res + Wc
#         else:
#             if 'str' in str(type(object_x[attr_idx])):
#                 print 'nononononono'
#             # numerical attribute
#             tmp = object_x[attr_idx] - object_y[attr_idx]
#             res = res + tmp * tmp
#     return res

def get_categorical_attributes_weight(df):
    '''  get categorical attribute weight by numerical attribute's standard deviation
        '''
    Wc = 0.0
    n = 0.0
    for attr_idx in xrange(0, df.shape[1]):
        if df.dtypes[attr_idx] != 'object':
            n = n + 1
            Wc = Wc + df.iloc[:,attr_idx].std()
    Wc = Wc / n
    return Wc

def k_prototype(n_cluster, max_iter, df):
    ''' k_prototype algorithm
        input:   1. the number of cluter
                    2. maximum iteration run into k_prototype
                    3. data in pandas dataframe format
        output: 1. k cluster in dataframe list (each dataframe represents a cluster)
                    2. k cluster_id according to dataframe
        '''
    cluster_id = defaultdict(list)                                   # cid -> objectid_list table
    res = []                                                         # dataframe list, clustering results
    global categorical_attributes_weight
    categorical_attributes_weight = [get_categorical_attributes_weight(df)] * n_cluster      # categorical data's weight

    # select k initial centroids
    centroid_id = random.sample(range(0, df.shape[0]), n_cluster)
    centroid = [df.iloc[x].copy() for x in centroid_id]

    for it in xrange(0, max_iter):
        if 'testing' not in __main__.__file__:
            sys.stdout.write("-> Iteration: %d   \r" % (it) )
            sys.stdout.flush()
        cluster_id.clear()

        # Assign all object to nearest centroid
        for object_id in xrange(0, df.shape[0]):
            min_dis = 1e9
            min_dis_cluster_id = -1
            for centroid_id, centroid_instance in enumerate(centroid):
                x = np.array(df.iloc[object_id])
                y = np.array(centroid_instance)

                tmp_distance = distance_fun.kp_distance(x, y, categorical_attributes_weight[centroid_id])
                if tmp_distance < min_dis:
                    min_dis = tmp_distance
                    min_dis_cluster_id = centroid_id
            cluster_id[min_dis_cluster_id].append(object_id)


        # Recompute centroid of each cluster
        for cid in xrange(0, n_cluster):
            # if a cluster is empty, borrow a object from the other cluster
            if len(cluster_id[cid]) <= 0:
                cluster_id[cid].append(cluster_id[ (cid + 1) % n_cluster][0])
                cluster_id[ (cid + 1) % n_cluster].pop(0)

            cluster_df = df.iloc[cluster_id[cid]]
            for attr_idx in xrange(0, cluster_df.shape[1]):
                # set centroid's attribute = mode or mean
                #if('str' in str(type(centroid[cid][attr_idx])) ):
                if(cluster_df.dtypes[attr_idx] == object):
                    centroid[cid][attr_idx] =  cluster_df[cluster_df.columns[attr_idx]].value_counts().idxmax()
                else:
                    centroid[cid][attr_idx] =  cluster_df[cluster_df.columns[attr_idx]].mean()

            # Recompute categorical attribute weight for this cluster
            categorical_attributes_weight[cid] = get_categorical_attributes_weight(cluster_df)

    # split original dataframe into n_cluster dataframe
    for c in xrange(0, n_cluster):
        res.append(df.iloc[cluster_id[c]])
    return res, cluster_id

def pair_categorical_values_distance(Ai, Aj, x, y):
    ''' compute the distance of Ai's  x and y with respect to Aj
        '''
    res = 0.0

    # pre calculation every posiible (ai) values + (ai, aj) pair occurence times
    counter = Counter(Ai)
    co_counter= Counter(zip(Ai, Aj))
    unique_values_in_Aj = np.unique(Aj)

    for aj_value in unique_values_in_Aj:
        # p( aj_value | x )
        pux = 0 if counter[x] == 0 else float(co_counter[(x, aj_value)]) / float(counter[x])

        # p(aj_value | y)
        puy = 0 if counter[y] == 0 else float(co_counter[(y, aj_value)]) / float(counter[y])
        res = res + pux if pux >= puy else res + puy
    res = res - 1
    return res

def get_all_categorical_attributes_values_distance(df):
    ''' Compute all pair-wise categorical values distance, return a dictionary back
        '''

    distance_table = {}
    counter = 1.0
    categorical_attributes_idx = [attr_idx for attr_idx in xrange(0, df.shape[1]) if not ('a' <= df.iloc[0, attr_idx] <= 'z')]
    ncategorical_attrbutes = float(len(categorical_attributes_idx))

    for attr_idx_i in categorical_attributes_idx:
        if not ('a' <= df.iloc[0, attr_idx_i] <= 'z'):
            if 'testing' not in __main__.__file__:
                #logger.debug("-> Compute categorical attribute distance: %f %%  \r" % (counter / ncategorical_attrbutes * 100))
                sys.stdout.write("-> Compute categorical attribute distance: %f %%  \r" % (counter / ncategorical_attrbutes * 100) )
                sys.stdout.flush()
                counter = counter + 1
            Ai = np.array(df[df.columns[attr_idx_i]] )
            unique_values_in_Ai = np.unique(Ai)
            unique_values_in_Ai = [x for x in unique_values_in_Ai if x != np.nan]

            #  enumerate all categorical pair value in Ai
            for i in xrange(0, len(unique_values_in_Ai)):
                for k in xrange(i + 1, len(unique_values_in_Ai)):
                    # initialize table value = 0
                    distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])] = distance_table[(unique_values_in_Ai[k], unique_values_in_Ai[i])] = 0.0

                    # enumerate other attributes Aj
                    for attr_idx_k in xrange(0, df.shape[1]):
                        if attr_idx_k != attr_idx_i :
                            Aj = np.array(df[df.columns[attr_idx_k]])
                            distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])] = distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])] \
                                + pair_categorical_values_distance(Ai, Aj, unique_values_in_Ai[i], unique_values_in_Ai[k])
                    distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])] = distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])] / (float(df.shape[1] - 1))
                    distance_table[(unique_values_in_Ai[k], unique_values_in_Ai[i])] = distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])]
                    #print str(unique_values_in_Ai[i]) + ' ' + str(unique_values_in_Ai[k]) + ' : ',
                    #print distance_table[(unique_values_in_Ai[i], unique_values_in_Ai[k])]

    return distance_table

def get_discretized_dataframe(df, ninterval):
    ''' discretize all numerical attributes and get discretized dataframe
           give categorical values from 'a' to 'z' in string type
        '''

    # normalize nuerical attribute from 0 to 1 for std_bound
    df = preprocessing.normalize_numerical_attributes(df)

    # bound points are from 0 to 1
    std_bound = [(1.0 / float(ninterval) ) * num for num in xrange(0, ninterval)]

    for column_id in xrange(0, df.shape[1]):
        if df.dtypes[column_id] != 'object':
            attr_discretization = (np.digitize(df[df.columns[column_id]], std_bound))
            attr_discretization = map(lambda x : chr(ord('a') + x - 1), attr_discretization)
            df[df.columns[column_id]] = df[df.columns[column_id]].astype(object)
            df[df.columns[column_id]] = attr_discretization
    return df

def get_all_numerical_attributes_weight(df):
    ''' compute the significance of all numerical attributes
           return a dictionary weight_table, key = attribute_index and value = significance
        '''

    if df.shape[1] == 1:
        return {0:1.0}
    weight_table = {}
    numerical_attributes_idx = [attr_idx for attr_idx in xrange(0, df.shape[1]) if ('a' <= df.iloc[0, attr_idx] <= 'z')]

    for attr_idx_i in numerical_attributes_idx:
        # init weight table
        weight_table[attr_idx_i] = 0.0
        values = np.unique(df[df.columns[attr_idx_i]])
        nvalues = len(values)

        # enumerate all categorical values
        for v1_idx in xrange(0, nvalues):
            for v2_idx in xrange(v1_idx + 1, nvalues):

                # get pair categorical values distance
                pair_distance = 0.0
                for attr_idx_k in xrange(0, df.shape[1]):
                    if attr_idx_k != attr_idx_i:
                        pair_distance = pair_distance + \
                                        pair_categorical_values_distance(np.array(df[df.columns[attr_idx_i]]), np.array(df[df.columns[attr_idx_k]]), str(values[v1_idx]), str(values[v2_idx]) )

                pair_distance = pair_distance / float(df.shape[1] - 1)
                weight_table[attr_idx_i] = weight_table[attr_idx_i] + pair_distance

        weight_table[attr_idx_i] = weight_table[attr_idx_i] / float(nvalues * (nvalues - 1) / 2)

    return weight_table

def mkm_center_and_instance_distance(center , instance):
    distance = 0.0
    for attr_id in xrange(0 , len(center)):
        #print str(center[attr_id]) + ': '  + str(type(center[attr_id]))
        #print str(instance[attr_id]) + ': '  + str(type(instance[attr_id]))
        if 'str' in str(type(center[attr_id])) :
            # categorical attributes distance
            attr_values = center[attr_id].split(',')

            for value in attr_values:
                #parsing ratio and categorical value
                ratio, categorical_value = value.split(':')
                #print categorical_value
                #print instance[attr_id]
                if categorical_value == instance[attr_id]:
                    distance = distance + 0.0
                else:
                    distance = distance + ((float(ratio) * (categorical_attributes_distance[categorical_value, instance[attr_id]]) ) ** 2)
        else:
            # numerical attributes distance
            tmp = numerical_attributes_weight[attr_id] * (center[attr_id] - instance[attr_id])
            distance = distance + tmp * tmp
    return distance

def modified_k_means_clustering(n_cluster, max_iter, df):
    ''' modified_k_mean_clustering
        input:   1. the number of cluster
                    2. maximum iteration run into modified_k_mean_clustering
                    3. data in pandas dataframe format
        output: 1. k cluster in dataframe list (each dataframe represents a cluster)
                    2. k cluster_id according to original dataframe
        '''

    cluster_id = defaultdict(list)            # cid -> objectid_list, each cluster's object id
    res = []                                  # dataframe list, clustering results

    # compute all categorical values distance
    df_discretized = get_discretized_dataframe(df.copy(), 5)
    global categorical_attributes_distance
    categorical_attributes_distance = get_all_categorical_attributes_values_distance(df_discretized)

    # compute numerical attribute significance
    global numerical_attributes_weight
    numerical_attributes_weight = get_all_numerical_attributes_weight(df_discretized)

    # select k initial centroids
    centroid_id = random.sample(range(0, df.shape[0]), n_cluster)
    centroid = [df.iloc[x].copy() for x in centroid_id]
    for centroid_instance in centroid:
        for attr_idx in xrange(0, df.shape[1]):
            if 'str' in str(type(centroid_instance[attr_idx]) ):
                centroid_instance[attr_idx] = '1.0:' + centroid_instance[attr_idx]

    # modified_k_mean_clustering
    for it in xrange(0, max_iter):
        if 'testing' not in __main__.__file__:
            sys.stdout.write("-> Iteration: %d   \r" % (it) )
            sys.stdout.flush()
        cluster_id.clear()

        # Assign all object to nearest centroid
        for object_id in xrange(0, df.shape[0]):
            min_dis = 1e9
            min_dis_cluster_id = -1
            instance = np.array(df.iloc[object_id])

            for centroid_id, centroid_instance in enumerate(centroid):
                # compute the distance between cluster_center and the instance
                center = np.array(centroid_instance)
                tmp_distance = mkm_center_and_instance_distance(center, instance)

                if tmp_distance < min_dis:
                    min_dis = tmp_distance
                    min_dis_cluster_id = centroid_id
            cluster_id[min_dis_cluster_id].append(object_id)

        # Recompute centroid of each cluster
        for c in xrange(0, n_cluster):
            cluster_df = df.iloc[cluster_id[c]]
            for attr_idx in xrange(0, cluster_df.shape[1]):
                if(cluster_df.dtypes[attr_idx] == 'object' ):
                    # store all  the (occurrance ratio + categoical value) splited by ','
                    # ex: 0.626740947075住,0.300835654596商,0.0473537604457其他,0.025069637883工
                    value_counts = cluster_df[cluster_df.columns[attr_idx]].value_counts()

                    total_number = float(sum(value_counts[:,]))
                    value = ''
                    for idx in xrange(0, value_counts.shape[0]):
                        value = value + str(value_counts[idx] / total_number) + ':' + value_counts.index[idx] + ','
                    value = value[:-1]
                    centroid[c][attr_idx] = value
                else:
                    centroid[c][attr_idx] = cluster_df[cluster_df.columns[attr_idx]].mean()

    # split original dataframe into n_cluster dataframe
    for c in xrange(0, n_cluster):
        res.append(df.iloc[cluster_id[c]])
    return res, cluster_id


def evaluate_clustering(df_list, normalized_df_list, alg_name):
    ''' Evaluate the goodness of clustering
        1. Mean 2. Median 3. C.V.
        4. WSS  5. BSS
        note: In modified_k_means_clustering, all_cluster_centroid's categorical attribute is just mode
        '''
    print '-> Evaluation Result:'

    cv = 0.0
    # Calculate all_clusters_centroid : 
    all_clusters_centroid = []
    concate_df = pd.concat(normalized_df_list)
    
    # calculate every attribute in all_clusters_centroid
    for attr_idx in xrange(0, concate_df.shape[1]):
        if(concate_df.dtypes[attr_idx] == 'object' ):
            # mode
            all_clusters_centroid.append(concate_df.iloc[:, attr_idx].value_counts().idxmax())
        else:
            # mean
            all_clusters_centroid.append(concate_df.iloc[:, attr_idx].mean())

    for c, df in enumerate(df_list):
        print '# Cluster ' + str(c + 1) + ': ' + '總價元',
        print 'Mean:' + str(int(df['總價元'].mean())) + ' ',
        print 'Median:' + str(int(df['總價元'].median())) + ' ',
        if df.shape[0] > 1:
            print 'Std:'  + str(int(df['總價元'].std())),
        print 'Coefficient of Variation: ' + str( round( df['總價元'].std() / df['總價元'].mean() * 100) )+ '%'
        cv = cv + round( (df['總價元'].std() / df['總價元'].mean()) * 100 )

    WSS = 0.0
    BSS = 0.0
    for cid, df in enumerate(normalized_df_list):
        # Calculate this cluster centroid
        centroid = []
        for attr_idx in xrange(0, df.shape[1]):
            if(df.dtypes[attr_idx] == 'object' ):
                if alg_name == 'k_prototype':
                    # mode
                    centroid.append(concate_df.iloc[:, attr_idx].value_counts().idxmax())
                elif alg_name == 'modified_k_means_clustering':
                    # (occurrance ratio + categoical value) splited by ','
                    # ex: 0.626740947075住,0.300835654596商,0.0473537604457其他,0.025069637883工
                    value_counts = concate_df.iloc[:, attr_idx].value_counts()
                    total_number = float(sum(value_counts[:,]))
                    value = ''
                    for idx in xrange(0, value_counts.shape[0]):
                        value = value + str(value_counts[idx] / total_number) + ':' + value_counts.index[idx] + ','
                    value = value[:-1]
                    centroid.append(value)
            else:
                # mean
                centroid.append(df.iloc[:, attr_idx].mean())

        # WSS
        for object_idx in xrange(0, df.shape[0]):
            x = np.array(centroid)
            y = np.array(df.iloc[object_idx])
            if alg_name == 'k_prototype':
                tmp_distance = distance_fun.kp_distance(x, y, categorical_attributes_weight[cid])
                WSS = WSS + tmp_distance * tmp_distance
            elif alg_name == 'modified_k_means_clustering':
                tmp_distance = mkm_center_and_instance_distance(x, y)
                WSS = WSS + tmp_distance * tmp_distance

        # BSS
        x = np.array(centroid)
        y = np.array(all_clusters_centroid)
        if alg_name == 'k_prototype':
            tmp_distance = distance_fun.kp_distance(x, y, categorical_attributes_weight[cid])
            BSS = BSS + df.shape[0] * tmp_distance * tmp_distance
        elif alg_name == 'modified_k_means_clustering':
            tmp_distance = mkm_center_and_instance_distance(x, y)
            BSS = BSS + df.shape[0] * tmp_distance * tmp_distance

    print '# WSS: ' + str(WSS)
    print '# BSS: ' + str(BSS)
    print '# WSS + BSS: ' + str(WSS + BSS)
    print '# C.V.: ' + str(cv / float(len(df_list))) + '%'
    print ''

    return WSS, BSS, cv / float(len(df_list))

def matching(distance_table):
    distance_table = [[1e10, 1, 2],
             [1, 2, 3],
             [1, 1, 1]]
    m = Munkres()
    indexes = m.compute(distance_table)
    total = 0
    for rid, cid in indexes:
        value = distance_table[rid][cid]
        total += value
        print '(cluster %d -> cluster %d\') cost =  %d' % (rid + 1, cid + 1, value)
    print 'total cost: %d' % total
    sys.exit(0)