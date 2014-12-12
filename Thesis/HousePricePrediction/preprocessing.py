# coding=utf-8
def feature_selection(df):
    ''' select the features
        '''
    df = df.drop(['車位類別', '非都市土地使用分區', '非都市土地使用編定', '土地區段位置或建物區門牌'], axis=1)
    df = df[ (df['交易標的'] != '土地') & (df['交易標的'] != '車位') & (df['交易標的'] != '建物')]
    return df

def fill_NaN(df):
    ''' fill mean or mode into NaN
        '''
    #df.fillna(df.mode().iloc[0])
    #df.fillna(df.mean())

    for attr_idx in xrange(0, df.shape[1]):
        if df.dtypes[attr_idx] == 'object':
            df[df.columns[attr_idx]] = df[df.columns[attr_idx]].replace(float('nan'), df[df.columns[attr_idx]].value_counts().idxmax())
        else :
            df[df.columns[attr_idx]] = df[df.columns[attr_idx]].replace(float('nan'), df[df.columns[attr_idx]].mean())
    return df

def normalize_numerical_attributes(df):
    ''' normalize numerical attributes
        '''
    for column_id in range(0, df.shape[1]):
        if df.dtypes[column_id] != object:
            shift = (df.iloc[:, column_id] - df.iloc[:, column_id].min())
            scale = (df.iloc[:, column_id].max() - df.iloc[:, column_id].min())
            df.iloc[:, column_id] = shift / scale
    return df