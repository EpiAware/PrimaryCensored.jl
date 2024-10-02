using Random
using Test
using TestItems
using Distributions
using PrimaryCensored

# Bad inputs
@testitem "Test struct type" begin
    using Distributions
    @test_throws MethodError within_interval_censored(Normal(1, 2), 7)
end

@testitem "Test constructor" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    use_dist_censored = within_interval_censored(use_dist, 3, 10)
    @test typeof(use_dist_censored) <: PrimaryCensored.WithinIntervalCensored
end

@testitem "Test random generation" begin
    using Distributions
    using Random
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    use_dist_trunc = truncated(use_dist, 3, 10)
    use_dist_censored = within_interval_censored(use_dist_trunc, 3, 10)
    rng = MersenneTwister(1234)
    @test length(rand(rng, use_dist_censored)) == 1

    out_rngs = zeros(1000000)
    for i in 1:1000000
        out_rngs[i] = rand(rng, use_dist_censored)
    end

    @test length(out_rngs) ≈ 1e6
    @test maximum(out_rngs) <= 10
    @test minimum(out_rngs) >= 0
end

@testitem "Test pdf method" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    use_dist_censored = within_interval_censored(use_dist, 3, 10)
    @test pdf(use_dist_censored) ≈ 0.157658245
    @test logpdf(use_dist_censored) ≈ log(0.157658245)
end
