import pandas as pd, numpy as np
from itertools import product


def normalize_dist(dist):
    return dist / dist.sum()


def update_prior(prior, likelihood):
    _prior = prior.copy()
    posterior = normalize_dist(_prior * likelihood)
    return posterior


def make_die(sides=6):
    return pd.Series(np.ones(sides)/sides, index=np.arange(1, sides+1))


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


_func_name = {
    'gt': 'greater',
    'gte': 'greater_equal',
    'lt': 'less',
    'lte': 'less_equal',
    'eq': 'equal',
}

def prob_superiority(pmf1, pmf2, method='gt'):
    qX, qY = np.meshgrid(pmf1.index.values, pmf2.index.values)
    pX, pY = np.meshgrid(pmf1, pmf2)
    p = (pX * pY * np.__dict__[_func_name[method]](qX, qY)).sum()
    return p

