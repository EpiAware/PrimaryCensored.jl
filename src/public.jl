# Public API declarations for Julia 1.11+

# Core distribution types (public but not exported)
public PrimaryCensored
public IntervalCensored
public Weighted
public Convolved

# Pluggable integration payload, re-exported from ConvolvedDistributions.
# Public but not exported so it never clashes with `Integrals.GaussLegendre`
# when both are loaded.
public GaussLegendre
