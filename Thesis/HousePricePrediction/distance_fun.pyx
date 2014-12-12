cimport numpy as np
cimport cython
import numpy as np
def kp_distance(np.ndarray object_x, np.ndarray object_y, float Wc):
    ''' Distance function for two objects
        '''
    cdef float res = 0.0
    cdef unsigned attr_idx
    for attr_idx in xrange(0, len(object_x)):
        cur_type = type(object_x[attr_idx])
        if( (cur_type == str) or (cur_type == np.string_) ):
            # categorical attribute
            if (object_x[attr_idx] == object_y[attr_idx]):
                res = res + Wc
        else:
            # numerical attribute
            tmp = object_x[attr_idx] - object_y[attr_idx]
            res = res + tmp * tmp
    return res
