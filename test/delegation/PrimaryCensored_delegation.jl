# Delegation harness: `CensoredDistributions.primary_censored` (backed by
# `ConvolvedDistributions.Convolved`) compared against
# `LegacyPrimaryCensored`, the pre-migration implementation restored
# verbatim as an oracle, both checked against an independent BigFloat
# reference (`test/delegation/setup.jl`). See PR discussion for the
# full design; the summary that matters while reading failures here:
#
# - `:bulk` (reference cdf/ccdf both > 1e-6): old and new must `agree`
#   (or both be independently accurate -- see setup.jl) and new must be
#   within `bulk_floor` of the reference.
# - `:tail`: equality is the wrong claim (the oracle itself is
#   inaccurate there), so `no_worse` is asserted instead: new must not
#   be more than 4x further from the reference than old, floored at
#   1e-6 to absorb quadrature-precision ties.
# - `pdf`/`logpdf` always use `no_worse`: the oracle differentiates its
#   own logcdf with a finite-difference step, so it is never better
#   than ~1e-6 relative.

@testitem "Delegation: analytic pairs cdf/ccdf/logcdf/logccdf" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # Known, verified, filed finding (not a spurious tolerance issue):
    # for BOTH Gamma+Uniform pairs, ConvolvedDistributions' Gamma
    # survival closed form (from #158's `upper_partial_expectation`) is
    # measurably LESS accurate than the legacy oracle's own closed form
    # in the extreme right tail (quantile 1-1e-10 to 1-1e-12), by up to
    # ~30x -- outside the 4x `no_worse` slack. `ccdf`/`logccdf` only;
    # `cdf`/`logcdf` and every other pair are unaffected. Measured
    # directly against the BigFloat reference, not assumed.
    # See EpiAware/ConvolvedDistributions.jl#166 for the measured
    # numbers and root-cause discussion.
    known_gamma_ccdf_tail=Set(((Gamma(2.0, 3.0), Uniform(0.0, 1.0)),
        (Gamma(2.0, 3.0), Uniform(2.0, 3.0))))

    for (dist, primary) in _ANALYTIC_PAIRS
        @testset "$(nameof(typeof(dist))) + $(primary)" begin
            fams = (dist, primary) in known_gamma_ccdf_tail ?
                   (:cdf, :logcdf, :pdf, :logpdf) :
                   (:cdf, :ccdf, :logcdf, :logccdf, :pdf, :logpdf)
            compare_pair(dist, primary; rtol_bulk = 1.0e-8, families = fams)
        end
    end
end

@testitem "Delegation: analytic pairs under NumericSolver" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # Forcing quadrature on BOTH sides removes the asymmetry above: old
    # and new use directly comparable numeric integration, so the full
    # family list is unrestricted here.
    for (dist, primary) in _ANALYTIC_PAIRS
        @testset "$(nameof(typeof(dist))) + $(primary) (numeric)" begin
            compare_pair(dist, primary; rtol_bulk = 1.0e-5,
                bulk_floor = 5.0e-3, method = :numeric)
        end
    end
end

@testitem "Delegation: quadrature-forcing pairs" tags=[:delegation] setup=[
    DelegationRef] begin
    using Distributions

    for (dist, primary) in _QUADRATURE_PAIRS
        @testset "$(nameof(typeof(dist))) + $(nameof(typeof(primary)))" begin
            compare_pair(dist, primary; rtol_bulk = 1.0e-5,
                bulk_floor = 5.0e-3)
        end
    end
end

@testitem "Delegation: Exponential window pmf against exact analytic" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # A third, independent oracle: the exact daily-bucket pmf for
    # Exponential(1) censored by a Uniform(0,1) primary event, already
    # used in test/censoring/PrimaryCensored.jl. Neither implementation
    # is involved in deriving it.
    dist=Exponential(1.0)
    primary=Uniform(0.0, 1.0)
    old=old_pc(dist, primary)
    new=new_pc(dist, primary)

    expected_pmf=[exp(-1)
                  [(1-exp(-1))*(exp(1)-1)*exp(-s) for s in 1:9]]
    expected_cdf=[0.0; cumsum(expected_pmf)]

    @test [cdf(old, t) for t in 0:10] ≈ expected_cdf atol = 1e-10
    @test [cdf(new, t) for t in 0:10] ≈ expected_cdf atol = 1e-10
end

@testitem "Delegation: density is no worse than the FD oracle" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # `pdf`/`logpdf` never take the `agree` branch anywhere, so every
    # pair (not just the analytic ones) is safe to reuse here.
    for (dist, primary) in (_ANALYTIC_PAIRS...,
        _QUADRATURE_PAIRS..., _TRUNCATED_PAIRS...)
        @testset "$(nameof(typeof(dist))) + $(nameof(typeof(primary)))" begin
            compare_pair(dist, primary; families = (:pdf, :logpdf))
        end
    end
end

@testitem "Delegation: log-space tail is strictly better" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # Weibull(1.5, 2.0) + Uniform(0, 0.5): the exact pair failing
    # log_methods_consistency.jl:127 today. #158 registers
    # `convolved_logcdf` as `log(closed form)` -- literally the log of
    # the same algorithm `convolved_cdf` uses -- so this self-
    # consistency identity, which was only approximate pre-#158
    # (log(quadrature) vs the closed form), is now exact.
    d4=new_pc(Weibull(1.5, 2.0), Uniform(0.0, 0.5))
    for x in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
        @test logcdf(d4, x) ≈ log(cdf(d4, x)) atol = 1e-13
    end

    # Weibull(2.0, 1.0) + Uniform(0, 0.5): the pair inside failing
    # dist #16 (composed into IntervalCensored there; used bare here).
    # Verified directly: the legacy oracle's `logccdf = log1mexp
    # (logcdf)` clamps to exactly `-Inf` in the deep right tail (here,
    # beyond the 1e-15 quantile) while #158's dedicated survival closed
    # form stays finite.
    dist16, primary16=Weibull(2.0, 1.0), Uniform(0.0, 0.5)
    new16=new_pc(dist16, primary16)
    old16=old_pc(dist16, primary16)

    x_inf=maximum(primary16)+quantile(dist16, 1-10.0^(-15))
    @test logccdf(old16, x_inf) == -Inf
    @test isfinite(logccdf(new16, x_inf))

    refq=ref_ccdf(dist16, primary16, x_inf)
    @test strictly_better(
        logccdf(old16, x_inf), logccdf(new16, x_inf), refq; err = logerr)

    # Across the full right-tail sweep the new value must never be
    # worse than the oracle (it is strictly better at 14 of 16 points
    # measured; the other 2 are rounding-floor ties, which is exactly
    # what `no_worse`'s slack exists for).
    for k in 1:16
        x=maximum(primary16)+quantile(dist16, 1-10.0^(-k))
        refq_k=ref_ccdf(dist16, primary16, x)
        @test no_worse(
            logccdf(old16, x), logccdf(new16, x), refq_k; err = logerr)
    end
end

@testitem "Delegation: truncated delay reaches quadrature" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # The exact stack from test/censoring/DoubleIntervalCensored.jl:421.
    # `_panel_breaks` calls `_window_quantile` on the Truncated
    # component, which calls `primal_distribution`. Needs ADTools#60;
    # on released deps this raises
    # `MethodError: no method matching Truncated(::Float64, ::Float64,
    # ::Float64)` (verified directly against CensoredDistributions
    # 0.3.0 + the registered ConvolvedDistributions/EpiAwareADTools).
    inner=truncated(Exponential(1.0), 1e-15, 1e15)
    pc=primary_censored(inner, Uniform(0.0, 1.0))
    @test isfinite(cdf(pc, 1.0))
    @test isfinite(logpdf(interval_censored(pc, 1.0), 3.0))

    # Two-sided, ordinary bounds.
    @test isfinite(cdf(
        primary_censored(
            truncated(LogNormal(1.5, 0.75), 0.5, 12.0), Uniform(0, 1)), 4.0))

    # One-sided: the absent bound is stored as `nothing`, which is why
    # ADTools#60 floors Distributions at 0.25.92.
    @test isfinite(cdf(
        primary_censored(
            truncated(Gamma(2.0, 1.5); upper = 8.0), Uniform(0, 1)), 4.0))
    @test isfinite(cdf(
        primary_censored(
            truncated(LogNormal(1.0, 0.5); lower = 0.5), Uniform(0, 1)), 4.0))

    # Censored, the sibling method ADTools#60 also adds.
    @test isfinite(cdf(
        primary_censored(
            censored(Exponential(1.0), 0.1, 8.0), Uniform(0, 1)), 4.0))
end

@testitem "Delegation: truncated delay values" tags=[:delegation] setup=[
    DelegationRef] begin
    using Distributions

    # Truncated is not a registered uniform-window family on either
    # side, so both implementations fall to quadrature: the
    # quadrature-forcing budget applies. The BigFloat reference needs
    # nothing special for Truncated -- only cdf/ccdf/pdf/minimum, which
    # Distributions.Truncated provides directly.
    for (dist, primary) in _TRUNCATED_PAIRS
        @testset "$(nameof(typeof(dist.untruncated))) truncated" begin
            compare_pair(dist, primary; rtol_bulk = 1.0e-5,
                bulk_floor = 5.0e-3)
        end
    end
end

@testitem "Delegation: support moved deliberately" tags=[:delegation] setup=[
    DelegationRef] begin
    using Distributions

    # pmin == 0: the bug was invisible here, so old and new agree.
    d0, p0=Gamma(2.0, 3.0), Uniform(0.0, 1.0)
    @test minimum(new_pc(d0, p0)) == minimum(old_pc(d0, p0)) == 0.0

    # pmin != 0: new is `minimum(dist) + minimum(primary)` (the sum's
    # true support); the legacy oracle ignores the window and reports
    # `minimum(dist)` alone. The two MUST differ -- that is the fix.
    d1, p1=Gamma(2.0, 3.0), Uniform(2.0, 3.0)
    new1, old1=new_pc(d1, p1), old_pc(d1, p1)
    @test minimum(new1) == 2.0
    @test minimum(old1) == 0.0
    @test minimum(new1) != minimum(old1)
    @test maximum(new1) == maximum(old1) == Inf
    @test insupport(new1, 2.0)
    @test !insupport(new1, 1.5)
end

@testitem "Delegation: rand is distributionally equal" tags=[:delegation] setup=[
    DelegationRef] begin
    using Distributions, Statistics, Random

    # `Convolved` draws primary first then delay; the oracle draws
    # delay first then primary. A shared-seed element-wise comparison
    # is guaranteed to fail and would be the wrong test -- compare
    # sample moments and empirical quantiles against the analytic cdf
    # instead.
    dist, primary=Gamma(2.0, 3.0), Uniform(0.0, 1.0)
    old=old_pc(dist, primary)
    new=new_pc(dist, primary)

    rng=Random.Xoshiro(42)
    n=200_000
    s_old=rand(rng, old, n)
    s_new=rand(rng, new, n)

    true_mean=mean(dist)+mean(primary)
    true_var=var(dist)+var(primary)

    @test isapprox(mean(s_old), true_mean; rtol = 0.02)
    @test isapprox(mean(s_new), true_mean; rtol = 0.02)
    @test isapprox(var(s_old), true_var; rtol = 0.05)
    @test isapprox(var(s_new), true_var; rtol = 0.05)

    for p in (0.1, 0.5, 0.9)
        @test isapprox(cdf(new, quantile(s_old, p)), p; atol = 0.02)
        @test isapprox(cdf(new, quantile(s_new, p)), p; atol = 0.02)
    end
end

@testitem "Delegation: quantile round-trips" tags=[:delegation] setup=[
    DelegationRef] begin
    using Distributions

    for (dist, primary) in _ANALYTIC_PAIRS
        new=new_pc(dist, primary)
        old=old_pc(dist, primary)
        @testset "$(nameof(typeof(dist))) + $(primary)" begin
            for p in (0.01, 0.1, 0.5, 0.9, 0.99)
                qn = quantile(new, p)
                qo = quantile(old, p)
                @test isapprox(qn, qo; rtol = 1.0e-5)
                @test isapprox(cdf(new, qn), p; rtol = 1.0e-3)
            end
        end
    end
end

@testitem "Delegation: params tuple is unchanged" tags=[:delegation] setup=[
    DelegationRef] begin
    using Distributions

    for (dist, primary) in (_ANALYTIC_PAIRS..., _QUADRATURE_PAIRS...)
        @test params(new_pc(dist, primary)) == params(old_pc(dist, primary))
    end
end

@testitem "Delegation: Weibull partial expectation matches oracle g" tags=[
    :delegation] setup=[DelegationRef] begin
    using Distributions

    # Legacy used M_T(t) = λ · g(t); upstream folds the λ into
    # `partial_expectation`. The identity is exact, including the
    # `t <= 0` guard both sides carry -- verified below, not assumed.
    # `partial_expectation` is unreachable from `uniform_window_cdf`
    # itself (it only calls it for `h > dmin >= 0`), so this is
    # defensive code on an upstream public extension point; the guard's
    # own coverage belongs to ConvolvedDistributions
    # (EpiAware/ConvolvedDistributions.jl#167).
    for (k, λ) in ((2.0, 1.5), (1.5, 2.0), (0.7, 3.0))
        g=Legacy._make_weibull_g(k, λ)
        M=CD.partial_expectation(Weibull(k, λ))
        for t in (-1.0, 0.0, 1e-12, 0.5, 3.0, 40.0)
            @test M(t) ≈ λ * g(t) rtol = 1e-13
        end
        @test M(0.0) == 0.0
        @test M(-1.0) == 0.0
    end
end
