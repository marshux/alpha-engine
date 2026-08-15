"""HMM-based market regime classifier: 4 latent regimes (BULL_TREND,
BEAR_TREND, CRISIS, LOW_VOL_GRIND) fitted on macro/market features,
producing a regime label per call plus two deterministic lookups derived
from that label -- a gross-exposure scalar and a set of signal-weight
overrides. Point-in-time safe: every call refits on features built from
data <= `as_of` only (see `build_regime_features`'s `lookback_days`
trailing window) -- no persisted model state between calls, unlike
`alpha_engine.ml.alpha_model.AlphaModel`, which *does* need retraining
discipline because a LightGBM fit is too slow to redo at every rebalance.
A GaussianHMM fit on a few years of daily macro data is fast enough to
just redo every call, so there's no staleness risk to manage here.

This sits alongside the existing Faber-style trend filter
(decision/risk.py::regime_is_risk_on) as an alternative regime mode
(strategy.yaml's `regime.mode: "trend_filter" | "hmm"`), not a
replacement for it -- the trend filter is a binary risk-on/risk-off
signal; this classifier is richer (regime *type*, not just direction) and
drives a continuous gross-exposure scalar rather than an all-or-nothing
flatten-to-cash. See backtest/engine.py and live/runner.py for how the
two modes are selected.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# hmmlearn logs a "Model is not converging" WARNING internally (not a
# Python `warnings.warn`, so `warnings.filterwarnings` can't touch it) on
# any fit that doesn't fully converge within `n_iter` -- routine on
# small/noisy real windows, not something a strategy or its operator
# needs to see every rebalance. `evaluate_regime`'s own try/except above
# is what actually handles a genuinely degenerate fit; this just quiets
# the expected non-convergence noise.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

REGIME_LABELS = ("BULL_TREND", "BEAR_TREND", "CRISIS", "LOW_VOL_GRIND", "TRANSITION")

# Guide section 3.6's REGIME_WEIGHTS table, mapped onto this repo's actual
# signal names (momentum/mean_reversion/volatility/value/quality/ml_alpha
# -- no ai_theme; strategies using the HMM regime mode carry none). Raw
# weights, not deltas -- applied wholesale to signal_weights.* via
# alpha_engine.config.apply_overrides, the same mechanism used for every
# other per-strategy config override in this codebase.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL_TREND": {
        "momentum": 0.30,
        "mean_reversion": 0.0,
        "volatility": 0.10,
        "value": 0.05,
        "quality": 0.15,
        "ml_alpha": 0.40,
    },
    "BEAR_TREND": {
        "momentum": 0.10,
        "mean_reversion": 0.0,
        "volatility": 0.20,
        "value": 0.20,
        "quality": 0.20,
        "ml_alpha": 0.30,
    },
    "CRISIS": {
        "momentum": 0.05,
        "mean_reversion": 0.0,
        "volatility": 0.30,
        "value": 0.10,
        "quality": 0.25,
        "ml_alpha": 0.30,
    },
    "LOW_VOL_GRIND": {
        "momentum": 0.20,
        "mean_reversion": 0.0,
        "volatility": 0.10,
        "value": 0.10,
        "quality": 0.15,
        "ml_alpha": 0.45,
    },
}

# Guide section 3.6's GROSS_EXPOSURE_BY_REGIME table -- illustrative
# starting points, not validated for this repo's universe/cost structure.
# Overridable per-strategy via strategy.yaml's regime.exposure_bull/
# regime.exposure_low_vol_grind/regime.exposure_bear/regime.exposure_crisis/
# regime.exposure_transition (see resolve_exposure_table below) -- e.g. for
# a walk-forward grid search over these floors via run_walk_forward's
# existing param_grid mechanism, the same way top_n/sizing are already
# tuned. TRANSITION's entry here (0.75) matches DEFAULT_GROSS_EXPOSURE --
# giving it an explicit, overridable entry (added for
# systematic_core_action_plan (1).md's Action 2, which grid-searches it
# alongside BEAR_TREND/CRISIS) rather than only ever falling through to
# the unrecognized-state fallback below, which stays hardcoded 0.75 on
# purpose -- a genuinely unrecognized label shouldn't be tunable the same
# way a real, named regime is.
GROSS_EXPOSURE_BY_REGIME: dict[str, float] = {
    "BULL_TREND": 1.0,
    "LOW_VOL_GRIND": 0.9,
    "BEAR_TREND": 0.60,
    "CRISIS": 0.35,
    "TRANSITION": 0.75,
}
DEFAULT_GROSS_EXPOSURE = 0.75  # unrecognized-state fallback

_EXPOSURE_CONFIG_KEYS = {
    "BULL_TREND": "exposure_bull",
    "LOW_VOL_GRIND": "exposure_low_vol_grind",
    "BEAR_TREND": "exposure_bear",
    "CRISIS": "exposure_crisis",
    "TRANSITION": "exposure_transition",
}


def resolve_exposure_table(regime_cfg: dict | None) -> dict[str, float]:
    """`strategy_config["regime"]`'s exposure_bull/exposure_low_vol_grind/
    exposure_bear/exposure_crisis/exposure_transition overrides layered on
    top of GROSS_EXPOSURE_BY_REGIME's defaults -- any key not present in
    `regime_cfg` keeps its default. Lets a strategy (or a walk-forward
    param_grid) tune these floors via the ordinary dotted-override
    mechanism (config.py::apply_overrides) instead of editing this
    module's constants directly.
    """
    regime_cfg = regime_cfg or {}
    return {
        label: regime_cfg.get(key, GROSS_EXPOSURE_BY_REGIME[label])
        for label, key in _EXPOSURE_CONFIG_KEYS.items()
    }


MIN_HISTORY_ROWS = 30  # below this, evaluate_regime degrades to the rule-based fallback rather than fitting


def build_regime_features(
    prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    macro_data: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 252 * 3,
) -> pd.DataFrame:
    """Daily feature panel ending at `as_of` (inclusive), up to
    `lookback_days` trading days long. Columns: spy_return_1m,
    spy_return_3m, realized_vol_1m, vix_level, vix_change_1m,
    yield_curve, credit_spread, breadth_200ma.

    Price/breadth-derived columns (spy_return_1m/3m, realized_vol_1m,
    breadth_200ma) only go NaN during a transient rolling-window warmup,
    so those rows are dropped -- same posture as every other signal
    module's warmup handling. Macro-derived columns are instead
    `fillna(0.0)`, NOT dropped: some FRED series have a genuine, lasting
    historical gap rather than a brief warmup -- found in practice with
    BAMLH0A0HYM2 (ICE BofA high-yield credit spread), whose free/
    anonymous CSV access only covers a rolling ~3-year window, not full
    history (a licensing constraint on that series specifically, not a
    fetch bug -- confirmed directly against FRED's endpoint; DGS10/DGS2/
    VIXCLS all return full history unrestricted). A blanket `.dropna()`
    across all 8 columns meant that one column's multi-year gap silently
    discarded the entire feature panel for every date in that gap --
    exactly the kind of single-missing-signal-nukes-everything failure
    this codebase's missing-data convention (see decision/scorer.py's
    combine_scores) otherwise always avoids. `fillna(0.0)` matches
    `alpha_engine.ml.features.build_ml_features`'s identical fallback for
    the same three macro columns, so both consumers of macro data degrade
    the same way.
    """
    bench = benchmark_prices.loc[:as_of]["close"]
    bench_ret = bench.pct_change()

    closes = pd.DataFrame({symbol: df["close"] for symbol, df in prices.items()}).loc[:as_of]
    above_200ma = closes.gt(closes.rolling(200).mean())
    breadth = above_200ma.mean(axis=1)

    macro = macro_data.loc[:as_of]

    price_derived = pd.DataFrame(
        {
            "spy_return_1m": bench.pct_change(21),
            "spy_return_3m": bench.pct_change(63),
            "realized_vol_1m": bench_ret.rolling(21).std() * np.sqrt(252),
            "breadth_200ma": breadth,
        }
    ).dropna()

    macro_derived = pd.DataFrame(
        {
            "vix_level": macro["vix"],
            "vix_change_1m": macro["vix"].pct_change(21),
            "yield_curve": macro["yield_curve"],
            "credit_spread": macro["credit_spread"],
        }
    ).reindex(price_derived.index).fillna(0.0)

    features = price_derived.join(macro_derived)
    return features.tail(lookback_days)


class RegimeClassifier:
    """Wraps a `GaussianHMM(n_components=4)`. `fit` and `predict_current`
    are always called together in practice (see `evaluate_regime` below)
    -- this class holds no state across separate as-of dates by design,
    since point-in-time safety requires refitting on each call's own
    truncated history anyway. `covariance_type="diag"` rather than the
    guide's "full": with 8 features x 4 components, a full covariance
    matrix is numerically fragile on the smaller early-history windows a
    walk-forward backtest will hit; diagonal is the standard, more
    robust choice for small-sample HMM fits.
    """

    def __init__(self, n_components: int = 4, random_state: int = 42, exposure_table: dict[str, float] | None = None):
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=random_state,
        )
        self._label_map: dict[int, str] = {}
        self._exposure_table = exposure_table or GROSS_EXPOSURE_BY_REGIME

    def fit(self, features: pd.DataFrame) -> None:
        if len(features) < MIN_HISTORY_ROWS:
            raise ValueError(
                f"need at least {MIN_HISTORY_ROWS} rows of history to fit a regime classifier, got {len(features)}"
            )
        self.model.fit(features.values)
        if _is_degenerate(self.model):
            # Proactive check right after fit(), rather than waiting for
            # hmmlearn's own internal _check() to raise partway through a
            # later .predict() call (both happen in practice -- see
            # _is_degenerate's docstring -- catching it here fails faster
            # and with a clearer message).
            raise ValueError("degenerate HMM fit: collapsed transition matrix, non-finite means, or near-zero variance")
        self._label_map = self._interpret_regimes(features)

    def predict_current(self, features: pd.DataFrame) -> str:
        state_idx = int(self.model.predict(features.values)[-1])
        return self._label_map.get(state_idx, "TRANSITION")

    def gross_exposure_for(self, label: str) -> float:
        return self._exposure_table.get(label, DEFAULT_GROSS_EXPOSURE)

    def weight_overrides_for(self, label: str) -> dict[str, float] | None:
        return REGIME_WEIGHTS.get(label)

    def _interpret_regimes(self, features: pd.DataFrame) -> dict[int, str]:
        """After fitting, label each latent state by inspecting its mean
        feature vector.

        Deliberately NOT the guide's illustrative version (fixed
        magnitude thresholds on *both* return and vol simultaneously,
        e.g. "BULL_TREND requires ret_1m > 1% AND vol_1m < 15%"): with
        only `n_components` states summarizing years of history, real
        state means rarely land inside a specific joint AND-region on
        two continuous dimensions at once, so most states fell through
        to a catch-all default far more often than intended -- found via
        real walk-forward testing, where the classifier landed on
        "ambiguous" for nearly an entire multi-year test window,
        including one that badly needed a real read. This version keeps
        only genuine extremes as absolute thresholds (CRISIS), and
        otherwise classifies on return *sign* alone, which is
        exhaustive -- every non-crisis state is unambiguously either
        net-positive or net-negative, so every state gets a real label
        with real consequences (BULL_TREND/LOW_VOL_GRIND on the
        upside, BEAR_TREND -- deliberately including a "quiet decline"
        that isn't high-vol enough to be CRISIS, not just choppy
        drops -- on the downside). Per-state "TRANSITION" is gone
        entirely; that label now only appears via evaluate_regime's
        outer insufficient-history/degenerate-fit fallback, a genuinely
        different situation (no information at all, vs. information
        that doesn't fit a bucket).
        """
        cols = list(features.columns)
        ret_1m_idx = cols.index("spy_return_1m")
        vol_1m_idx = cols.index("realized_vol_1m")
        vix_idx = cols.index("vix_level")

        labels: dict[int, str] = {}
        for state_idx, mean in enumerate(self.model.means_):
            ret_1m, vol_1m, vix = mean[ret_1m_idx], mean[vol_1m_idx], mean[vix_idx]
            if vix > 30 or vol_1m > 0.30:
                labels[state_idx] = "CRISIS"
            elif ret_1m > 0 and vol_1m < 0.12:
                labels[state_idx] = "LOW_VOL_GRIND"
            elif ret_1m > 0:
                labels[state_idx] = "BULL_TREND"
            else:
                labels[state_idx] = "BEAR_TREND"
        return labels


def _is_degenerate(model: GaussianHMM) -> bool:
    """True if a `GaussianHMM.fit()` produced unusable parameters: a
    transition row that never accumulated any probability mass (a state
    the Viterbi path never actually visits), non-finite means, or a
    collapsed (near-zero) emission variance in any state/feature.
    `GaussianHMM` with only 4 components fit on a modest real dataset
    hits this in practice, and left unchecked it doesn't fail until a
    *later* `.predict()` call raises deep inside hmmlearn's internals
    (`ValueError: startprob_ must sum to 1 (got nan)` /
    `transmat_ rows must sum to 1`) -- checking explicitly right after
    fit() catches it immediately with a clear cause instead.
    """
    if not np.isfinite(model.transmat_).all() or (model.transmat_.sum(axis=1) < 0.99).any():
        return True
    if not np.isfinite(model.means_).all():
        return True
    # model.covars_ is always returned as a full (n_components,
    # n_features, n_features) matrix regardless of covariance_type (see
    # hmmlearn's `fill_covars`) -- for covariance_type="diag" the
    # off-diagonal entries are structurally zero by construction, not a
    # sign of degeneracy, so only the diagonal (the actual per-feature
    # variances) is checked here.
    variances = np.diagonal(model.covars_, axis1=-2, axis2=-1)
    if not np.isfinite(variances).all() or (variances < 1e-8).any():
        return True
    return False


def _simple_regime_fallback(benchmark_prices: pd.DataFrame, macro_data: pd.DataFrame, as_of: pd.Timestamp) -> str:
    """Rule-based regime read requiring no fitting at all: VIX level plus
    price-vs-200-day-moving-average, the same two signals the guide's
    macro/regime signals (section 3.3's `vix_regime`, decision/risk.py's
    existing Faber trend filter) already rely on individually. Used
    whenever the HMM can't produce a usable read (too little history, or
    a degenerate fit -- see `_is_degenerate`) instead of a constant
    default: a constant default provides zero regime-sensitivity for
    whatever period it covers, while this rule -- simple as it is --
    still reacts to genuinely stressed markets. Never itself raises: at
    worst (no benchmark history yet) it falls through to "TRANSITION",
    same conservative-not-aggressive posture as everywhere else in this
    module.
    """
    bench = benchmark_prices.loc[:as_of]["close"]
    macro = macro_data.loc[:as_of]
    vix = None
    if not macro.empty and pd.notna(macro["vix"].iloc[-1]):
        vix = float(macro["vix"].iloc[-1])

    if len(bench) < 200:
        return "TRANSITION"
    spy_vs_200ma = float(bench.iloc[-1] / bench.rolling(200).mean().iloc[-1] - 1.0)

    if (vix is not None and vix > 30) or spy_vs_200ma < -0.15:
        return "CRISIS"
    if (vix is not None and vix > 22) or spy_vs_200ma < -0.05:
        return "BEAR_TREND"
    if vix is not None and vix < 15 and spy_vs_200ma > 0.05:
        return "BULL_TREND"
    if spy_vs_200ma > 0:
        return "LOW_VOL_GRIND"
    return "TRANSITION"


def evaluate_regime(
    prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    macro_data: pd.DataFrame,
    as_of: pd.Timestamp,
    exposure_table: dict[str, float] | None = None,
) -> tuple[str, float, dict[str, float] | None]:
    """Convenience wrapper used by backtest/engine.py and live/runner.py:
    build features, fit, and classify the current (as_of) regime in one
    call. Returns (label, gross_exposure_fraction, weight_overrides).

    `exposure_table`, if given: overrides `GROSS_EXPOSURE_BY_REGIME` (see
    `resolve_exposure_table`) -- lets a strategy's config tune the
    exposure floors without editing this module.

    Falls back to `_simple_regime_fallback`'s rule-based read (see its
    docstring) if there isn't yet enough history to fit (e.g. very early
    in a backtest window) or if the fit itself produces a degenerate
    model (`_is_degenerate`) -- same "degrade gracefully rather than
    fail" posture every other missing-data case in this codebase already
    follows, but preferring a real (if simple) regime read over a
    constant one wherever there's enough raw price/VIX data to compute
    it at all.
    """
    exposure_table = exposure_table or GROSS_EXPOSURE_BY_REGIME
    features = build_regime_features(prices, benchmark_prices, macro_data, as_of)
    if len(features) < MIN_HISTORY_ROWS:
        label = _simple_regime_fallback(benchmark_prices, macro_data, as_of)
        return label, exposure_table.get(label, DEFAULT_GROSS_EXPOSURE), REGIME_WEIGHTS.get(label)

    try:
        classifier = RegimeClassifier(exposure_table=exposure_table)
        classifier.fit(features)
        label = classifier.predict_current(features)
        return label, classifier.gross_exposure_for(label), classifier.weight_overrides_for(label)
    except Exception:
        logger.warning(
            "Regime classifier fit/predict failed for as_of=%s -- using rule-based fallback", as_of, exc_info=True
        )
        label = _simple_regime_fallback(benchmark_prices, macro_data, as_of)
        return label, exposure_table.get(label, DEFAULT_GROSS_EXPOSURE), REGIME_WEIGHTS.get(label)
