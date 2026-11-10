import unittest
import numpy as np
from aurum.statistics import simultaneous_intervals


class PairedStratifiedInferenceTests(unittest.TestCase):
    def test_source_conditional_means_and_se(self):
        rng=np.random.default_rng(9)
        strata=np.repeat(['arxiv','wiki'],[211,389]);values=rng.normal(size=600)
        arxiv=strata=='arxiv';wiki=~arxiv
        family=np.c_[values,values*arxiv/arxiv.mean(),values*wiki/wiki.mean()]
        result=simultaneous_intervals(family,strata,draws=2000)
        np.testing.assert_allclose(result['mean'],[values.mean(),values[arxiv].mean(),values[wiki].mean()],atol=1e-14)
        np.testing.assert_allclose(result['se'][1:],[values[arxiv].std(ddof=1)/np.sqrt(arxiv.sum()),
                                                   values[wiki].std(ddof=1)/np.sqrt(wiki.sum())],atol=1e-14)

    def test_identical_and_degenerate_contrasts(self):
        rng=np.random.default_rng(8);x=rng.normal(size=400);strata=np.repeat(['a','b'],200)
        result=simultaneous_intervals(np.c_[x,x,np.zeros(400)],strata,draws=5000)
        self.assertEqual(result['lower'][0],result['lower'][1])
        self.assertEqual(result['upper'][0],result['upper'][1])
        self.assertEqual(result['lower'][2],0);self.assertEqual(result['upper'][2],0)
        self.assertGreater(result['critical'],1.8);self.assertLess(result['critical'],2.1)

    def test_stratified_standard_error(self):
        rng=np.random.default_rng(17);strata=np.repeat(['a','b'],[180,220]);x=rng.normal(size=400)
        result=simultaneous_intervals(x,strata,draws=1000)
        expected=np.sqrt(sum((sum(strata==s)/400)**2*x[strata==s].var(ddof=1)/sum(strata==s) for s in ['a','b']))
        self.assertAlmostEqual(result['se'][0],expected,places=14)


if __name__=='__main__':unittest.main()
