# Shared machinery for the delegation harness. Three things live here:
# the two constructors under comparison, an independent BigFloat
# reference that goes through neither of them, and the two assertion
# predicates (`agree` for the regime where both are accurate, `no_worse`
# for the regime where the oracle is not).
#
# The reference is written for an arbitrary continuous `primary`, not
# only `Uniform`: `F_S(x) = ∫ f_primary(p) F_dist(x - p) dp` over
# `primary`'s support. For a `Uniform` primary this reduces to the
# windowed form `(1/w) ∫_{h-w}^{h} F_dist(u) du`, so nothing is lost for
# the uniform-window pairs, and the `Beta` primary pair (no closed form
# on either side) gets the same treatment instead of a special case.

@testsnippet DelegationRef begin
    using Distributions
    using Statistics
    using Random
    using FastGaussQuadrature: gausslegendre
    using CensoredDistributions
    import LegacyPrimaryCensored as Legacy
    import ConvolvedDistributions as CD

    # --- shared pair lists ------------------------------------------------
    #
    # `@testitem`s referencing this setup get these by plain name (same
    # mechanism as the functions below); TestItemRunner statically scans
    # for `@testitem`/`@testsnippet` and never executes other top-level
    # code in a test file, so these cannot live at file scope outside a
    # snippet.

    const _ANALYTIC_PAIRS = (
        (Gamma(2.0, 3.0), Uniform(0.0, 1.0)),
        (LogNormal(1.5, 0.75), Uniform(0.0, 1.0)),
        (Weibull(2.0, 1.5), Uniform(0.0, 1.0)),
        (Gamma(2.0, 3.0), Uniform(2.0, 3.0)),
        (Weibull(1.5, 2.0), Uniform(0.0, 0.5)),
        (Weibull(2.0, 1.0), Uniform(0.0, 0.5)),
        (LogNormal(1.0, 0.5), Uniform(0.5, 1.5)))

    const _QUADRATURE_PAIRS = (
        (Exponential(1.0), Uniform(0.0, 1.0)),
        (Exponential(2.0), Uniform(0.0, 2.0)),
        (Gamma(2.0, 3.0), Beta(2.0, 2.0)))

    const _TRUNCATED_PAIRS = (
        (truncated(Exponential(1.0), 1e-15, 1e15), Uniform(0.0, 1.0)),
        (truncated(LogNormal(1.5, 0.75), 0.5, 12.0), Uniform(0.0, 1.0)),
        (truncated(Gamma(2.0, 1.5); upper = 8.0), Uniform(0.0, 1.0)),
        (truncated(LogNormal(1.0, 0.5); lower = 0.5), Uniform(0.0, 1.0)))

    # --- the two objects under comparison ------------------------------

    new_pc(dist,
        primary;
        method = nothing) = method === nothing ? primary_censored(dist, primary) :
                            primary_censored(dist, primary; method = method)

    # The oracle's solver types are distinct from the production ones, so
    # a caller passes `:analytic` / `:numeric` rather than an object.
    function old_pc(dist, primary; method = nothing)
        method === nothing &&
            return Legacy.legacy_primary_censored(dist, primary)
        m = method === :numeric ? Legacy.NumericSolver() :
            Legacy.AnalyticalSolver()
        return Legacy.legacy_primary_censored(dist, primary; method = m)
    end

    prod_method(sym) = sym === :numeric ? NumericSolver() : AnalyticalSolver()

    # --- independent reference -------------------------------------------
    #
    # Every integrand below is positive, so a composite rule accumulated
    # in BigFloat keeps full RELATIVE accuracy exactly where the closed
    # form's cancelling-terms formula does not. That makes this the
    # ground truth in the tail; the oracle is not.

    const _REF_N, _REF_W = gausslegendre(40)
    const _REF_PANELS = 64

    function _panelled(f, lo::BigFloat, hi::BigFloat)
        hi <= lo && return zero(BigFloat)
        acc = zero(BigFloat)
        step = (hi - lo) / _REF_PANELS
        half = step / 2
        for p in 1:_REF_PANELS
            mid = lo + (p - 1) * step + half
            s = zero(BigFloat)
            for i in eachindex(_REF_N)
                s += BigFloat(_REF_W[i]) * f(mid + half * BigFloat(_REF_N[i]))
            end
            acc += half * s
        end
        return acc
    end

    function ref_cdf(dist, primary, x::Real)
        plo, phi = BigFloat(minimum(primary)), BigFloat(maximum(primary))
        return _panelled(
            p -> BigFloat(pdf(primary, Float64(p))) *
                 BigFloat(cdf(dist, Float64(BigFloat(x) - p))),
            plo, phi)
    end

    function ref_ccdf(dist, primary, x::Real)
        plo, phi = BigFloat(minimum(primary)), BigFloat(maximum(primary))
        return _panelled(
            p -> BigFloat(pdf(primary, Float64(p))) *
                 BigFloat(ccdf(dist, Float64(BigFloat(x) - p))),
            plo, phi)
    end

    function ref_pdf(dist, primary, x::Real)
        plo, phi = BigFloat(minimum(primary)), BigFloat(maximum(primary))
        return _panelled(
            p -> BigFloat(pdf(primary, Float64(p))) *
                 BigFloat(pdf(dist, Float64(BigFloat(x) - p))),
            plo, phi)
    end

    # --- error measures -------------------------------------------------

    function relerr(v::Real, ref::BigFloat)
        ref == 0 && return v == 0 ? 0.0 : Inf
        isnan(v) && return Inf
        return Float64(abs(BigFloat(v) - ref) / abs(ref))
    end

    # An absolute gap in logs IS a relative gap in linear space, so the
    # same budget applies. `-Inf` against a positive reference is total
    # loss, which is exactly the failure CD#158 removes. When the
    # reference itself is Float64-zero (the true density underflows,
    # which happens 1 ULP inside a support boundary), any v so negative
    # that `exp(v)` also underflows is an equally valid "zero" answer,
    # not worse than a literal `-Inf` — otherwise a finite-but-tiny
    # value like -758.9 is scored as total loss while `-Inf` scores
    # perfectly, backwards from what either number is claiming.
    function logerr(v::Real, ref::BigFloat)
        ref <= 0 && return (v == -Inf || exp(v) == 0.0) ? 0.0 : Inf
        (isnan(v) || !isfinite(v)) && return Inf
        return Float64(abs(BigFloat(v) - log(ref)))
    end

    # --- assertion predicates -------------------------------------------

    # Regime 1: both implementations are inside their accurate range, so
    # the migration must not have moved the number. `rtol` is an
    # arithmetic-reassociation budget, not a correctness budget.
    agree(old, new; rtol) = old == new || isapprox(old, new; rtol = rtol)

    # Regime 2: the oracle is itself wrong, so equality is the wrong
    # assertion. Require the new value to be no further from the
    # reference than the oracle. The slack absorbs the case where both
    # sit at the rounding floor and ordering is arbitrary; the floor
    # stops a spurious failure when the oracle happens to be exact by
    # luck.
    const NO_WORSE_SLACK = 4.0
    # Measured, not guessed: two independent 64-node Gauss-Legendre
    # quadratures (the oracle's single panel, upstream's
    # quantile-panelled rule) agree with the BigFloat reference to
    # ~1e-7-1e-8 relative on `pdf`, and at that level which one happens
    # to be closer is quadrature-node-placement noise, not a
    # correctness signal — e.g. Gamma(2,3)+Uniform(0,1) at x=0.274 has
    # both sides accurate to ~1e-7 but a 5.5x ratio between them. 1e-6
    # absorbs that without masking any of the genuine multi-order-of-
    # magnitude regressions the harness does catch (e.g. the deep
    # right-tail `ccdf`/`logccdf` finding on the same pair, both errors
    # already > 1e-6).
    const NO_WORSE_FLOOR = 1.0e-6

    no_worse(old,
        new,
        ref;
        err = relerr) = err(new, ref) <= max(NO_WORSE_SLACK * err(old, ref), NO_WORSE_FLOOR)

    # The claim CD#158 makes, stated as a claim rather than smuggled in
    # under the slack factor.
    strictly_better(old, new, ref; err = relerr) = err(new, ref) < err(old, ref)

    # Classified from the reference alone, so it never depends on the
    # code under test.
    #
    # Threshold calibrated by measurement, not guessed: at a reference
    # value of ~1e-9 the uniform-window closed form (both the oracle's
    # and upstream's, since they share the cancelling `h·F(h) - l·F(l)
    # - (M(h) - M(l))` shape) already carries ~1e-13 absolute error from
    # subtracting mean-sized terms, which is a ~1e-4 RELATIVE error at
    # that magnitude — an order of magnitude past `bulk_floor` before
    # the naive 1e-10 split would call it :bulk. 1e-6 keeps the
    # remaining bulk region several orders of magnitude clear of that
    # floor (absolute cancellation error / 1e-6 ≈ 1e-7 minimum safe
    # reference), while everything below it correctly falls to the
    # `no_worse` tail comparison.
    regime(refc::BigFloat, refq::BigFloat) = (refc > 1.0e-6 && refq > 1.0e-6) ? :bulk :
                                             :tail

    # --- x grids ----------------------------------------------------------
    #
    # 50 bulk points reproduce the sweep the deleted
    # `test/censoring/primarycensored_cdf.jl` ran. At that density two to
    # three points land strictly inside the first window, so both arms of
    # `l > dmin` and the join at `x = a + w` are exercised; upstream's
    # seven fixed points can sit entirely on one side of that branch.

    bulk_grid(primary) = collect(range(minimum(primary) + 0.1, minimum(primary) + 20.0;
        length = 50))

    function boundary_grid(dist, primary)
        a = minimum(primary)
        w = maximum(primary) - a
        dmin = minimum(dist)
        pts = [a + dmin, a + w, prevfloat(a + w), nextfloat(a + w),
            a + dmin + w]
        return filter(isfinite, pts)
    end

    function left_tail_grid(dist, primary)
        base = minimum(primary) + (isfinite(minimum(dist)) ?
                                   minimum(dist) : 0.0)
        return [base + 10.0^e for e in range(-8, 0.5; length = 25)]
    end

    function right_tail_grid(dist, primary)
        a = minimum(primary)
        qs = [quantile(dist, 1 - 10.0^(-k)) for k in 1:12]
        return [a + q for q in qs if isfinite(q)]
    end

    function eval_grid(dist, primary)
        xs = vcat(bulk_grid(primary), boundary_grid(dist, primary),
            left_tail_grid(dist, primary), right_tail_grid(dist, primary))
        return sort!(unique!(filter(isfinite, xs)))
    end

    # --- the sweep ----------------------------------------------------------
    #
    # `rtol_bulk` is 1e-8 for pairs where both sides use the uniform-window
    # closed form and 1e-5 where both fall to quadrature: the oracle
    # integrates on a single 64-node Gauss-Legendre panel and the new code
    # on upstream's quantile-panelled rule, so they differ by construction.
    #
    # `pdf`/`logpdf` never take the `agree` branch: the oracle
    # differentiates its own logcdf with a 1e-8 step, so it is never
    # better than ~1e-6 relative and equality would be wrong at every x.

    # `families` restricts which of the six functions are asserted, so a
    # testitem can isolate e.g. just the density family without
    # re-asserting the cdf family it shares a grid with.
    function compare_pair(dist, primary; rtol_bulk = 1.0e-8,
            method = nothing, bulk_floor = 1.0e-6,
            families = (:cdf, :ccdf, :logcdf, :logccdf, :pdf, :logpdf))
        old = old_pc(dist, primary; method = method)
        new = new_pc(dist, primary;
            method = method === nothing ? nothing : prod_method(method))

        cdf_families = ((:cdf, cdf, :bulk_or_tail, relerr),
            (:ccdf, ccdf, :bulk_or_tail, relerr),
            (:logcdf, logcdf, :bulk_or_tail, logerr),
            (:logccdf, logccdf, :bulk_or_tail, logerr))

        for x in eval_grid(dist, primary)
            refc = ref_cdf(dist, primary, x)
            refq = ref_ccdf(dist, primary, x)
            refp = ref_pdf(dist, primary, x)
            r = regime(refc, refq)

            # Distribution-function family: agree in the bulk, no worse
            # in the tail.
            for (name, f, _, err) in cdf_families
                name in families || continue
                ref = name in (:cdf, :logcdf) ? refc : refq
                o, n = f(old, x), f(new, x)
                if r === :bulk
                    # `agree` (reassociation budget) OR both sides are
                    # independently within `bulk_floor` of the ground
                    # truth: the oracle's own closed form is not
                    # perfectly accurate even in :bulk (e.g. ~2e-8 on
                    # LogNormal+Uniform(0,1) ccdf), so a migration that
                    # makes the new value MORE accurate can legitimately
                    # separate from the oracle by a hair over
                    # `rtol_bulk`. That is success, not a failure to
                    # paper over.
                    both_accurate = err(n, ref) <= bulk_floor &&
                                    err(o, ref) <= bulk_floor
                    @test agree(o, n; rtol = rtol_bulk) || both_accurate
                    @test err(n, ref) <= bulk_floor
                else
                    @test no_worse(o, n, ref; err = err)
                end
            end

            # Density family: reference-only, both regimes.
            if :pdf in families
                @test no_worse(pdf(old, x), pdf(new, x), refp)
            end
            if :logpdf in families
                @test no_worse(logpdf(old, x), logpdf(new, x), refp;
                    err = logerr)
            end
        end
        return nothing
    end
end
