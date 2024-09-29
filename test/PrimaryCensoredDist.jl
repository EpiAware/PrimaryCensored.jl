using Random
using Test
using TestItems
using Distributions
using PrimaryCensored

# Bad inputs
@testitem "Test struct type" begin
    using Distributions
    @test_throws MethodError primarycensored(Normal(1, 2), 7)
end

@testitem "Test constructor" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test typeof(use_dist) <: PrimaryCensored.PrimaryCensoredDist
end

@testitem "Test random generation" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test length(rand(use_dist, 10)) == 10

    use_dist_trunc = truncated(use_dist, 3, 10)
    use_dist_trunc_rn = rand(use_dist_trunc, 1000000)
    @test length(use_dist_trunc_rn) ≈ 1e6
    @test maximum(use_dist_trunc_rn) <= 10
    @test minimum(use_dist_trunc_rn) >= 3
end

@tesitem "Params calls" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    extracted_params = params(use_dist)
    @test length(extracted_params) == 4
    @test extracted_params[1] ≈ 3.5
    @test extracted_params[3] ≈ 1.0
end

@testitem "Test cdf method" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test cdf(use_dist, 1e8) ≈ 1.0
    @test logcdf(use_dist, -Inf) ≈ -Inf
end

@testitem "Test ccdf" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test ccdf(use_dist, 1e8) ≈ 0.0
    @test ccdf(use_dist, 0.0) ≈ 1
    @test logccdf(use_dist, 0.0) ≈ 0.0
end
