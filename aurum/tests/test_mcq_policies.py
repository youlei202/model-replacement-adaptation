import itertools
import unittest
import pandas as pd
from aurum.mcq_analysis import policies


class CompleteFamilyPolicyTests(unittest.TestCase):
    def table(self):
        rows=[]
        for i,pair in enumerate(itertools.combinations(range(1,6),2)):
            for size,loss,cheap in [(60,.2 if i==0 else .3 if i==1 else .4,True),
                                    (400,.19 if i==0 else .1 if i==1 else .35,False)]:
                rows.append({'task':'synthetic','target_index':0,'pair':','.join(map(str,pair)),
                             'candidate_id':f'{i}_{size}','actual_bytes':size,'validation_tv':loss,
                             'family':'anchor' if cheap else 'linear','codec':'fp32','cheap':cheap})
        return pd.DataFrame(rows)

    def test_fixed_pair_retunes_and_positive_minimum(self):
        decisions,summary=policies(self.table(),[128,512])
        self.assertAlmostEqual(summary.iloc[0].Gamma_development,.045)
        fixed=decisions[decisions.policy.eq('fixed_pair_1_2')].sort_values('cap_bytes')
        self.assertEqual(fixed.actual_bytes.tolist(),[60,400])
        self.assertAlmostEqual(fixed.validation_tv.mean(),.195)
        self.assertEqual(decisions[decisions.policy.eq('joint')].pair.tolist(),['1,2','1,3'])

    def test_incomplete_family_rejected(self):
        table=self.table();table=table[table.pair.ne('4,5')]
        with self.assertRaises(ValueError):policies(table,[128,512])

    def test_stable_winner_zero_price(self):
        table=self.table();table.loc[table.pair.eq('1,2'),'validation_tv']=.05
        _,summary=policies(table,[128,512]);self.assertEqual(summary.iloc[0].Gamma_development,0.)


if __name__=='__main__':unittest.main()
