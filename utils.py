import pandas as pd, numpy as np
from itertools import product
from scipy.stats import gaussian_kde


def normalize_dist(dist):
    return dist / dist.sum()


def update_prior(prior, likelihood):
    posterior = normalize_dist(prior * likelihood)
    return posterior


def make_uniform_dist(qs):
    try:
        size = len(qs)
    except TypeError:
        qs = np.arange(0, qs)
        size = qs.size

    dist = pd.Series(np.ones(size), index=qs) / size
    return dist


def make_die(sides=6):
    return make_uniform_dist(np.arange(1, sides+1))


def add_dist(*dists):
    """Make an addends distribution."""
    res = {}
    for idxs in product(*[dist.index.values for dist in dists]):
        q = sum(idxs)
        p = 1
        for n, i in enumerate(idxs):
            p *= dists[n].loc[i] 
        
        res[q] = res.get(q, 0) + p

    pmfAdd = pd.Series(res).sort_index()
    return pmfAdd


def mix_dist(*dists, ps=None):
    """Make a mixture of distributions."""
    df = pd.DataFrame(dists).fillna(0)
    
    if ps is None:
        _n = len(dists)
        ps = np.ones(_n) / _n
    
    pmf = pd.DataFrame.mul(df, ps, axis='index').sum(0)
    return pmf


def cdf_to_pmf(cdf):
    _cdf = cdf.diff()
    _cdf.fillna(1-_cdf.sum(), inplace=True)
    return _cdf


_np_func_name = {
    'gt': 'greater',
    'gte': 'greater_equal',
    'lt': 'less',
    'lte': 'less_equal',
    'eq': 'equal',
}

def prob_superiority(pmf1, pmf2, method='gt'):
    qX, qY = np.meshgrid(pmf1.index.values, pmf2.index.values)
    pX, pY = np.meshgrid(pmf1, pmf2)
    p = (pX * pY * np.__dict__[_np_func_name[method]](qX, qY)).sum()
    return p


def fit_gaussian_kde_pmf(sample, qs):
    kde = gaussian_kde(sample)
    pmf = kde(qs)
    pmf = pd.Series(normalize_dist(pmf), index=qs)
    return pmf


def dist_mean(dist):
    mu = (dist.index.values * dist.values).sum()
    return mu


def dist_moment(dist, moment=2):
    mu = dist_mean(dist)
    m = np.sum((dist.index.values - mu) ** moment * dist.values) ** (1/moment)
    return m
