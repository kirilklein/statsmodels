"""
Created on Feb 3, 2022 1:04:22 PM

Author: Josef Perktold
License: BSD-3
"""
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.treatment.treatment_effects import TreatmentEffect

from .results import results_teffects as res_st

cur_dir = Path(__file__).parent.resolve()

file_name = "cataneo2.csv"
file_path = Path(cur_dir).joinpath("results", file_name)

dta_cat = pd.read_csv(file_path)

formula = "mbsmoke_ ~ mmarried_ + mage + mage2 + fbaby_ + medu"
res_probit = Probit.from_formula(formula, dta_cat).fit()

methods = [
    ("ra", res_st.results_ra),
    ("ipw", res_st.results_ipw),
    ("aipw", res_st.results_aipw),
    ("aipw_wls", res_st.results_aipw_wls),
    ("ipw_ra", res_st.results_ipwra),
    ]

method_labels = [
    ("ra", "RA"),
    ("ipw", "IPW"),
    ("aipw", "AIPW"),
    ("aipw_wls", "AIPW-WLS"),
    ("ipw_ra", "IPW-RA"),
    ]


class TestTEffects:

    @classmethod
    def setup_class(cls):
        formula_outcome = "bweight ~ prenatal1_ + mmarried_ + mage + fbaby_"
        mod = OLS.from_formula(formula_outcome, dta_cat)
        tind = np.asarray(dta_cat["mbsmoke_"])
        cls.teff = TreatmentEffect(mod, tind, results_select=res_probit)

    def test_aux(self):
        prob = res_probit.predict()
        assert prob.shape == (4642,)

    @pytest.mark.parametrize("case", method_labels)
    def test_method_label(self, case):
        # each estimator must label its own results, not report "IPW"
        meth, label = case
        res = getattr(self.teff, meth)(return_results=True)
        assert res.method == label

    @pytest.mark.parametrize("case", methods)
    def test_effects(self, case):
        meth, res2 = case
        teff = self.teff

        res1 = getattr(teff, meth)(return_results=False)
        assert_allclose(res1[:2], res2.table[:2, 0], rtol=1e-4)

        # if meth in ["ipw", "aipw", "aipw_wls", "ra", "ipw_ra"]:
        res0 = getattr(teff, meth)(return_results=True)
        assert_allclose(res1, res0.effect, rtol=1e-4)
        res1 = res0.results_gmm
        # TODO: check ra and ipw difference 5e-6, others pass at 1e-12
        assert_allclose(res0.start_params, res1.params, rtol=1e-5)
        assert_allclose(res1.params[:2], res2.table[:2, 0], rtol=1e-5)
        assert_allclose(res1.bse[:2], res2.table[:2, 1], rtol=1e-3)
        assert_allclose(res1.tvalues[:2], res2.table[:2, 2], rtol=1e-3)
        assert_allclose(res1.pvalues[:2], res2.table[:2, 3],
                        rtol=1e-4, atol=1e-15)
        ci = res1.conf_int()
        assert_allclose(ci[:2, 0], res2.table[:2, 4], rtol=5e-4)
        assert_allclose(ci[:2, 1], res2.table[:2, 5], rtol=5e-4)

        # test all GMM params
        # constant is in different position in Stata, `idx` rearanges
        k_p = len(res1.params)
        if k_p == 8:
            # IPW, no outcome regression
            idx = [0, 1, 7, 2, 3, 4, 5, 6]
        elif k_p == 18:
            idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10, 17, 12, 13, 14,
                   15, 16]
        elif k_p == 12:
            # RA, no selection regression
            idx = [0, 1, 6, 2, 3, 4, 5, 11, 7, 8, 9, 10]
        else:
            idx = np.arange(k_p)

        # TODO: check if improved optimization brings values closer
        assert_allclose(res1.params, res2.table[idx, 0], rtol=1e-4)
        assert_allclose(res1.bse, res2.table[idx, 1], rtol=0.05)

        # test effects on the treated, not available for aipw
        if not meth.startswith("aipw"):
            table = res2.table_t

            res1 = getattr(teff, meth)(return_results=False, effect_group=1)
            assert_allclose(res1[:2], table[:2, 0], rtol=1e-4)

            res0 = getattr(teff, meth)(return_results=True, effect_group=1)
            # TODO: check ipw difference 1e-5, others pass at 1e-12
            assert_allclose(res1, res0.effect, rtol=2e-5)
            res1 = res0.results_gmm
            # TODO: check ra difference 4e-5, others pass at 1e-12
            assert_allclose(res0.start_params, res1.params, rtol=5e-5)
            assert_allclose(res1.params[:2], table[:2, 0], rtol=5e-5)
            assert_allclose(res1.bse[:2], table[:2, 1], rtol=1e-3)
            assert_allclose(res1.tvalues[:2], table[:2, 2], rtol=1e-3)
            assert_allclose(res1.pvalues[:2], table[:2, 3],
                            rtol=1e-4, atol=1e-15)
            ci = res1.conf_int()
            assert_allclose(ci[:2, 0], table[:2, 4], rtol=5e-4)
            assert_allclose(ci[:2, 1], table[:2, 5], rtol=5e-4)

            # consistency check, effect on untreated,  not in Stata
            res1 = getattr(teff, meth)(return_results=False, effect_group=0)
            res0 = getattr(teff, meth)(return_results=True, effect_group=0)
            assert_allclose(res1, res0.effect, rtol=1e-12)
            assert_allclose(res0.start_params, res0.results_gmm.params,
                            rtol=1e-12)


@pytest.mark.parametrize("meth", ["ipw_ra", "aipw_wls"])
def test_select_params_not_six(meth):
    # GMM moment conditions used to hardcode 6 selection parameters
    formula_sel = "mbsmoke_ ~ mmarried_ + mage + fbaby_"
    res_sel = Probit.from_formula(formula_sel, dta_cat).fit(disp=0)
    formula_outcome = "bweight ~ prenatal1_ + mmarried_ + mage + fbaby_"
    mod = OLS.from_formula(formula_outcome, dta_cat)
    tind = np.asarray(dta_cat["mbsmoke_"])
    teff = TreatmentEffect(mod, tind, results_select=res_sel)

    res1 = getattr(teff, meth)(return_results=False)
    res0 = getattr(teff, meth)(return_results=True)
    assert_allclose(res1, res0.effect, rtol=1e-12)
    assert_allclose(res0.start_params, res0.results_gmm.params, rtol=1e-12)


class TestTEffectsGLM:
    # binary outcome with logit outcome model, no Stata reference values

    @classmethod
    def setup_class(cls):
        formula_outcome = "lbweight ~ prenatal1_ + mmarried_ + mage + fbaby_"
        mod = GLM.from_formula(formula_outcome, dta_cat,
                               family=families.Binomial())
        cls.tind = np.asarray(dta_cat["mbsmoke_"])
        cls.teff = TreatmentEffect(mod, cls.tind, results_select=res_probit)

    def test_family(self):
        assert isinstance(self.teff.results0.model.family, families.Binomial)
        assert isinstance(self.teff.results1.model.family, families.Binomial)

    @pytest.mark.parametrize("meth", ["ra", "ipw", "aipw", "aipw_wls",
                                      "ipw_ra"])
    def test_consistency(self, meth):
        res1 = getattr(self.teff, meth)(return_results=False)
        res0 = getattr(self.teff, meth)(return_results=True)
        assert_allclose(res1, res0.effect, rtol=1e-12)
        assert_allclose(res0.start_params, res0.results_gmm.params,
                        rtol=1e-12)
        assert np.all((res0.effect[1:] > 0) & (res0.effect[1:] < 1))

    def test_ra_gcomputation(self):
        # RA with a GLM outcome model is g-computation
        exog = self.teff.model_pool.exog
        y = self.teff.model_pool.endog
        pom = [GLM(y[self.tind == k], exog[self.tind == k],
                   family=families.Binomial()).fit().predict(exog).mean()
               for k in (0, 1)]
        _, pom0, pom1 = self.teff.ra(return_results=False)
        assert_allclose([pom0, pom1], pom, rtol=1e-10)

    def test_logit_outcome_model(self):
        # discrete Logit as outcome model gives the same RA as GLM Binomial
        mod = Logit.from_formula(
            "lbweight ~ prenatal1_ + mmarried_ + mage + fbaby_", dta_cat)
        res = TreatmentEffect(mod, self.tind, results_select=res_probit).ra()
        res_glm = self.teff.ra()
        assert_allclose(res.effect, res_glm.effect, rtol=1e-8)
        assert_allclose(res.sd, res_glm.sd, rtol=1e-6)

    def test_ipw_outcome_model_invariant(self):
        mod_ols = OLS.from_formula(
            "lbweight ~ prenatal1_ + mmarried_ + mage + fbaby_", dta_cat)
        res_ols = TreatmentEffect(mod_ols, self.tind,
                                  results_select=res_probit).ipw()
        res = self.teff.ipw()
        assert_allclose(res.effect, res_ols.effect, rtol=1e-12)
        assert_allclose(res.sd, res_ols.sd, rtol=1e-8)


def test_glm_ra_influence_function():
    # RA with logit outcome model: compare GMM standard errors of POM, ATE,
    # risk ratio and odds ratio with the analytic influence function
    # (delta method for g-computation)
    formula_outcome = "lbweight ~ prenatal1_ + mmarried_ + mage + fbaby_"
    mod = GLM.from_formula(formula_outcome, dta_cat,
                           family=families.Binomial())
    tind = np.asarray(dta_cat["mbsmoke_"])
    res = TreatmentEffect(mod, tind, results_select=res_probit).ra()

    y, x = mod.endog, mod.exog
    nobs = len(y)
    pom, infl = [], []
    for k in (0, 1):
        mask = tind == k
        mu = GLM(y[mask], x[mask], family=families.Binomial()).fit().predict(x)
        pom.append(mu.mean())
        score = np.zeros_like(x)
        score[mask] = (y[mask] - mu[mask])[:, None] * x[mask]
        hess = -(x[mask] * (mu[mask] * (1 - mu[mask]))[:, None]).T @ x[mask]
        jac = (x * (mu * (1 - mu))[:, None]).mean(0)
        infl.append((mu - pom[-1]) / nobs - score @ np.linalg.solve(hess, jac))
    infl = np.column_stack(infl)
    cov = infl.T @ infl
    m0, m1 = pom
    var_ate = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    assert_allclose(res.effect, [m1 - m0, m0, m1], rtol=1e-10)
    assert_allclose(res.sd, np.sqrt([var_ate, cov[0, 0], cov[1, 1]]),
                    rtol=1e-5)

    rr = res.risk_ratio()
    grad = np.array([-m1 / m0**2, 1 / m0])
    assert_allclose(rr.predicted(), m1 / m0, rtol=1e-10)
    assert_allclose(rr.se_vectorized(), np.sqrt(grad @ cov @ grad), rtol=1e-5)

    odds = res.odds_ratio()
    or_ = m1 / (1 - m1) / (m0 / (1 - m0))
    grad = np.array([-or_ / (m0 * (1 - m0)), or_ / (m1 * (1 - m1))])
    assert_allclose(odds.predicted(), or_, rtol=1e-10)
    assert_allclose(odds.se_vectorized(), np.sqrt(grad @ cov @ grad),
                    rtol=1e-5)
