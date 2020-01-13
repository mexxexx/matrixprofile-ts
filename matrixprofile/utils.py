# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np
import numpy.fft as fft

def zNormalize(ts):
    """
    Returns a z-normalized version of a time series.

    Parameters
    ----------
    ts: Time series to be normalized
    """

    ts -= np.mean(ts)
    std = np.std(ts)

    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")
    else:
        ts /= std

    return ts

def zNormalizeEuclidian(tsA,tsB):
    """
    Returns the z-normalized Euclidian distance between two time series.

    Parameters
    ----------
    tsA: Time series #1
    tsB: Time series #2
    """

    if len(tsA) != len(tsB):
        raise ValueError("tsA and tsB must be the same length")

    return np.linalg.norm(zNormalize(tsA.astype("float64")) - zNormalize(tsB.astype("float64")))

def np_rolling(a, window_size, f):
    nrows = a.size - window_size + 1
    n = a.strides[0]
    a2D = np.lib.stride_tricks.as_strided(a,shape=(nrows, window_size),strides=(n,n))
    return f(a2D,1)

def movmeanstd(ts,m):
    """
    Calculate the mean and standard deviation within a moving window passing across a time series.

    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")

    movmean = np_rolling(ts, m, np.mean)
    movstd = np_rolling(ts, m, np.std)

    return [movmean,movstd]

def movstd(ts,m):
    """
    Calculate the standard deviation within a moving window passing across a time series.

    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    
    movstd = np_rolling(ts, m, np.std)

    return movstd

def preprocess_ts(ts, m):
    """
    Calculates the FFT of a time series and returns some stats

    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.

    Returns
    -------
    X: the FFT
    n: length of the time series
    meanx: moving mean
    sigmax: moving std
    """

    n = len(ts)
    X = np.fft.fft(ts)

    meanx = np_rolling(ts, m, np.mean)
    sigmax = np_rolling(ts, m, np.std)

    return (X, n, meanx, sigmax)

def slidingDotProduct(query,ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.

    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """

    m = len(query)
    n = len(ts)


    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]


    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))


    #Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]

def DotProductStomp(ts,m,dot_first,dot_prev,order):
    """
    Updates the sliding dot product for a time series ts from the previous dot product dot_prev.

    Parameters
    ----------
    ts: Time series under analysis.
    m: Length of query within sliding dot product.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    order: The location of the first point in the query.
    """

    l = len(ts)-m+1
    dot = np.roll(dot_prev,1)

    dot += ts[order+m-1]*ts[m-1:l+m]-ts[order-1]*np.roll(ts[:l],1)

    #Update the first value in the dot product array
    dot[0] = dot_first[order]

    return dot


def mass(query, ts, noise_var=None):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS): a Euclidian distance similarity search algorithm. Note that we are returning the square of MASS.

    Parameters
    ----------
    query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
    ts: Time series to compare against query.
    noise_var: Variance of Gaussian noise overlying the signal. If no value is passed, no noise correction is applied.
    """

    #query_normalized = zNormalize(np.copy(query))
    m = len(query)
    X, n, meanx, sigmax = preprocess_ts(ts, m)

    res = massPreprocessed(query, X, n, m, meanx, sigmax, noise_var)
    return res

def massPreprocessed(query, X, n, m, meanx, sigmax, noise_var=None):
    """ 
    Returns the distance profile of a query within tsA against the time series tsB using the more efficient MASS comparison, where the time series is already transformed to FFT space.
    
    Parameters
    ----------
    query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
    X: FFT of time series to compare against query.
    n: length of time series
    m: length of query
    meanx: moving mean of time series
    sigmax: moving std of time series
    noise_var: Variance of Gaussian noise overlying the signal. If no value is passed, no noise correction is applied.
    """

    q_mean = np.mean(query)
    q_std = np.std(query)

    # reverse the query
    y = np.flip(query, 0)
   
    # make y same size as ts with zero fill
    y = np.concatenate([y, np.zeros(n-m)])

    # main trick of getting dot product in O(n log n) time
    Y = np.fft.fft(y)
    Z = X * Y
    z = np.fft.ifft(Z) # Z-normalize distances
    dist = np.empty(n-m+1)
    dist[:] = m 
    
    sigmax_zero = np.isclose(sigmax, 0)
    sigmax[sigmax_zero] = 1e-10 # avoid divide by 0

    # Handling of constant subsequences: No z-Normalization is possible in that case, but we can still shift by the mean. Then the distance squared between a z normalized u and a constant sequence is exactly m (because m = 1/std^2 * dist^2(u - mean)). If both sequences are constant, the distance is zero.
    if not np.isclose(q_std, 0): 
        dist = (z[m - 1:n] - m * meanx * q_mean)
        dist = m - dist / (sigmax * q_std)
        dist = np.real(2 * dist)

        if not noise_var is None:
            # Assuming that the signal is disturbed by gaussian noise, De Paepe et al propose a noise canceling algorithm that is implemented here.
            # For further reference see 'Eliminating Noise in the Matrix Profile' by Dieter De Paepe, Oliver Janssens and Sofie Van Hoecke published in ICPRAM 2019, DOI:10.5220/0007314100830093
            dist = np.maximum(0, dist - (2 + 2*m) * np.true_divide(noise_var, np.multiply(np.maximum(sigmax, q_std), np.maximum(sigmax, q_std))))
        dist[sigmax_zero] = m
    else:
        dist[sigmax_zero] = 0

    return np.sqrt(np.absolute(dist))

def massStomp(query,ts,dot_first,dot_prev,index,mean,std):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries using the STOMP dot product speedup. Note that we are returning the square of MASS.

    Parameters
    ----------
    query: Time series snippet to evaluate. Note that, for STOMP, the query must be a subset of ts.
    ts: Time series to compare against query.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    index: The location of the first point in the query.
    mean: Array containing the mean of every subsequence in ts.
    std: Array containing the standard deviation of every subsequence in ts.
    """
    m = len(query)
    dot = DotProductStomp(ts,m,dot_first,dot_prev,index)

    #Return both the MASS calcuation and the dot product
    res = 2*m*(1-(dot-m*mean[index]*mean)/(m*std[index]*std))

    return res, dot


def apply_av(mp,av=[1.0]):
    """
    Applies an annotation vector to a Matrix Profile.

    Parameters
    ----------
    mp: Tuple containing the Matrix Profile and Matrix Profile Index.
    av: Numpy array containing the annotation vector.
    """

    if len(mp[0]) != len(av):
        raise ValueError(
            "Annotation Vector must be the same length as the matrix profile")
    else:
        av_max = np.max(av)
        av_min = np.min(av)
        if av_max > 1 or av_min < 0:
            raise ValueError("Annotation Vector must be between 0 and 1")
        mp_corrected = mp[0] + (1 - np.array(av)) * np.max(mp[0])
        return (mp_corrected, mp[1])


def is_self_join(tsA, tsB):
    """
    Helper function to determine if a self join is occurring or not. When tsA 
    is absolutely equal to tsB, a self join is occurring.

    Parameters
    ----------
    tsA: Primary time series.
    tsB: Subquery time series.
    """
    return tsB is None or np.array_equal(tsA, tsB)
