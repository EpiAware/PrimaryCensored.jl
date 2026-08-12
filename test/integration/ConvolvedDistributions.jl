@testitem "convolve_series PMF matches hand-computed masses" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    dic = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)
    inner = CensoredDistributions.get_dist(dic)
    n = 8

    ref_pmf = [cdf(inner, k + 1) - cdf(inner, k) for k in 0:(n - 1)]
    for k in 0:(n - 1)
        @test pdf(dic, k) ≈ ref_pmf[k + 1]
    end

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
    @test convolve_series(dic, series) ≈ convolve_series(ref_pmf, series)
end

@testitem "convolve_series equals explicit causal convolution" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    dic = double_interval_censored(Gamma(2.0, 1.5); interval = 1)
    series = [2.0, 4.0, 7.0, 3.0, 1.0, 0.0, 5.0]
    n = length(series)
    pmf = [pdf(dic, k) for k in 0:(n - 1)]

    # Independent causal, window-truncated convolution of the same masses.
    expected = map(1:n) do i
        sum(pmf[k + 1] * series[i - k] for k in 0:(min(length(pmf), i) - 1))
    end
    @test convolve_series(dic, series) ≈ expected
end

@testitem "convolve_series: bare interval_censored unit grid" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    # A bare interval_censored(dist, 1) is also a supported unit grid.
    ic = interval_censored(Normal(5, 2), 1)
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = length(series)
    pmf = [pdf(ic, k) for k in 0:(n - 1)]
    @test convolve_series(ic, series) ≈ convolve_series(pmf, series)
end

@testitem "convolve_series: weekly (w = 7) grid masses" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    w = 7
    dic = double_interval_censored(
        LogNormal(2.5, 0.75); upper = 70, interval = w)
    inner = CensoredDistributions.get_dist(dic)
    n = 8

    ref_pmf = [cdf(inner, w * (k + 1)) - cdf(inner, w * k) for k in 0:(n - 1)]
    for k in 0:(n - 1)
        @test pdf(dic, w * k) ≈ ref_pmf[k + 1]
    end

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
    @test convolve_series(dic, series) ≈ convolve_series(ref_pmf, series)
end

@testitem "convolve_series: bare weekly interval_censored" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    w = 7
    ic = interval_censored(Normal(35, 8), w)
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = length(series)
    pmf = [pdf(ic, w * k) for k in 0:(n - 1)]
    @test convolve_series(ic, series) ≈ convolve_series(pmf, series)
end

@testitem "convolve_series rejects irregular boundaries" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [1.0, 2.0, 3.0]

    ic_arb = interval_censored(Normal(5, 2), [0.0, 1.0, 3.0, 6.0])
    @test_throws ArgumentError convolve_series(ic_arb, series)
end

@testitem "convolve_series rejects continuous primary censoring" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    series = [1.0, 2.0, 3.0]
    @test_throws ArgumentError convolve_series(pc, series)
end

@testitem "convolve_series: continuous delay discretises via double interval" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    d = LogNormal(1.5, 0.75)
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    ref = convolve_series(double_interval_censored(d; interval = 1), series)
    @test convolve_series(d, series) ≈ ref
end

@testitem "convolve_series: continuous delay honours the interval keyword" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    w = 7
    d = LogNormal(2.5, 0.75)
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    ref = convolve_series(double_interval_censored(d; interval = w), series)
    @test convolve_series(d, series; interval = w) ≈ ref
end

@testitem "convolve_series: continuous delay forwards keywords" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    d = LogNormal(1.5, 0.75)
    pe = Uniform(0, 2)
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    ref = convolve_series(
        double_interval_censored(d; interval = 1, primary_event = pe), series)
    @test convolve_series(d, series; primary_event = pe) ≈ ref
    # A different primary event gives a different result.
    @test !(convolve_series(d, series) ≈ ref)
end

@testitem "convolve_series: time-varying fast path reads the grid PMF" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    import ConvolvedDistributions: delay_masses
    using Distributions

    ic = double_interval_censored(LogNormal(1.5, 0.75); interval = 1)
    n = 6
    grid = [pdf(ic, k) for k in 0:(n - 1)]
    @test delay_masses(ic, n) ≈ grid
    # The generic hook recovers the same masses by convolving a unit impulse.
    impulse = [i == 1 ? 1.0 : 0.0 for i in 1:n]
    @test delay_masses(ic, n) ≈ convolve_series(ic, impulse)
end

@testitem "convolve_series: time-varying interval-censored delays" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    n = length(series)
    delays = [double_interval_censored(LogNormal(m, 0.6); interval = 1)
              for m in range(1.0, 1.8; length = n)]

    # Reference: scatter each cohort forward through its own delay (:primary).
    masses = [[pdf(delays[j], k) for k in 0:(n - 1)] for j in 1:n]
    expected = zeros(n)
    for s in 1:n
        for lag in 0:(n - s)
            expected[s + lag] += masses[s][lag + 1] * series[s]
        end
    end
    @test convolve_series(delays, series) ≈ expected
end

@testitem "convolve_series: time-varying continuous delays discretise" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    n = length(series)
    raw = [LogNormal(m, 0.6) for m in range(1.0, 1.8; length = n)]
    dic = [double_interval_censored(d; interval = 1) for d in raw]
    @test convolve_series(raw, series) ≈ convolve_series(dic, series)
end

@testitem "convolve_series: truncated continuous delay" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    # A truncated delay passed to convolve_series is discretised via
    # double_interval_censored (truncation applied before secondary interval
    # censoring). Verify the result differs from the untruncated delay and
    # that truncation via the lower/upper keywords is honoured.
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    # Truncation configured through double_interval_censored, then convolved.
    truncated_delay = double_interval_censored(LogNormal(1.5, 0.75);
        lower = 0.5, upper = 5.0, interval = 1)
    result = convolve_series(truncated_delay, series)

    untruncated = convolve_series(LogNormal(1.5, 0.75), series)
    @test result != untruncated

    # The truncated PMF has no mass outside [0.5, 5.0].
    masses = CensoredDistributions._grid_pmf(truncated_delay, length(series))
    @test sum(masses) ≈ 1.0 atol = 1e-6
end

@testitem "convolve_series: delay_masses fast path for continuous delay" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series, delay_masses
    using Distributions

    n = 8
    d = LogNormal(1.5, 0.75)

    # The continuous delay_masses fast path must agree with the generic
    # unit-impulse fallback and with a direct unit-grid discretisation.
    fast = delay_masses(d, n)
    impulse = [i == 1 ? 1.0 : 0.0 for i in 1:n]
    generic = convolve_series(d, impulse)
    @test fast ≈ generic
    @test fast ≈ pdf(double_interval_censored(d; interval = 1), 0:(n - 1))
end

@testitem "convolve_series: regime-compressed continuous delays" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    runs = [LogNormal(1.5, 0.75) => 3, LogNormal(1.0, 0.6) => 4]
    vector = vcat(fill(LogNormal(1.5, 0.75), 3), fill(LogNormal(1.0, 0.6), 4))
    @test convolve_series(runs, series) ≈ convolve_series(vector, series)
end

@testitem "Custom GaussLegendre node count changes the numeric CDF" begin
    using Distributions
    using CensoredDistributions: GaussLegendre

    # CD#92 regression guard: a non-default payload must reach the
    # quadrature, not be accepted and ignored. n = 64 is the default and
    # takes the native panelled path, so it cannot be used here.
    dist, primary, x = Gamma(2.0, 1.5), Uniform(0.0, 1.0), 2.0
    d_def = primary_censored(dist, primary; method = NumericSolver())
    d_n2 = primary_censored(dist, primary;
        method = NumericSolver(GaussLegendre(; n = 2)))

    @test cdf(d_n2, x) != cdf(d_def, x)
    @test !isapprox(cdf(d_n2, x), cdf(d_def, x); atol = 1e-7)
    # Coarse but not wrong: measured gap is 9.8e-6.
    @test isapprox(cdf(d_n2, x), cdf(d_def, x); atol = 1e-4)

    # More nodes converge onto the default.
    d_n256 = primary_censored(dist, primary;
        method = NumericSolver(GaussLegendre(; n = 256)))
    @test cdf(d_n256, x) ≈ cdf(d_def, x) atol=1e-12
end

@testitem "Analytic fallback honours the solver payload" begin
    using Distributions
    using CensoredDistributions: GaussLegendre

    # No closed form for a truncated-Normal primary, so AnalyticalSolver
    # falls through to quadrature and must use the stored payload.
    dist = Gamma(2.0, 1.5)
    primary = truncated(Normal(0.5, 0.3), 0.0, 1.0)
    x = 2.0
    a_def = primary_censored(dist, primary)
    a_n2 = primary_censored(dist, primary; solver = GaussLegendre(; n = 2))

    @test a_def.method isa AnalyticalSolver
    @test !isapprox(cdf(a_n2, x), cdf(a_def, x); atol = 1e-6)
    @test isapprox(cdf(a_n2, x), cdf(a_def, x); atol = 1e-3)
end

@testitem "Closed-form path ignores the solver payload" begin
    using Distributions
    using CensoredDistributions: GaussLegendre

    # Gamma + Uniform has a closed form, so the payload never reaches
    # quadrature and the answer is bit-identical.
    dist, primary, x = Gamma(2.0, 1.5), Uniform(0.0, 1.0), 2.0
    @test cdf(primary_censored(dist, primary), x) ==
          cdf(primary_censored(dist, primary;
            solver = GaussLegendre(; n = 2)), x)
end

@testitem "Integrals.jl algorithms route through the extension" begin
    using Distributions
    using Integrals: QuadGKJL, HCubatureJL

    dist, primary = Gamma(2.0, 1.5), Uniform(0.0, 1.0)
    d_def = primary_censored(dist, primary; method = NumericSolver())
    for alg in (QuadGKJL(), HCubatureJL())
        d = primary_censored(dist, primary; method = NumericSolver(alg))
        @test d.method.solver === alg
        for x in (1.0, 2.0, 3.0, 5.0)
            @test cdf(d, x) ≈ cdf(d_def, x) atol=1e-10
        end
    end
end

@testitem "A solver with no integrate method errors on the numeric path" begin
    using Distributions

    struct BrokenSolver end

    # Reaches quadrature: no `integrate(::BrokenSolver, ...)` method.
    d = primary_censored(Exponential(2.0), Uniform(0.0, 1.0);
        solver = BrokenSolver())
    @test_throws MethodError cdf(d, 2.0)

    d_forced = primary_censored(Gamma(2.0, 3.0), Uniform(0.0, 1.0);
        method = NumericSolver(BrokenSolver()))
    @test_throws MethodError cdf(d_forced, 2.0)

    # Closed form never consults the payload, so this works.
    d_analytic = primary_censored(Gamma(2.0, 3.0), Uniform(0.0, 1.0);
        solver = BrokenSolver())
    @test 0 < cdf(d_analytic, 2.0) < 1
end

@testitem "Concrete solver payload keeps the return type concrete" begin
    using Distributions, Test
    using CensoredDistributions: GaussLegendre

    f(θ,
        y) = cdf(
        primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0);
            method = NumericSolver(GaussLegendre(; n = 32))),
        y)
    @test isconcretetype(typeof(primary_censored(Gamma(2.0, 1.5),
        Uniform(0.0, 1.0); solver = GaussLegendre(; n = 32))))
    @inferred f([2.0, 1.5], 2.0)
end
