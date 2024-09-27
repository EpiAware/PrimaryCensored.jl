using Random
using Test
using TestItems
using Distributions
using PrimaryCensored

# Bad inputs
@testitem "Test struct type" begin
    using Distributions
    @test_throws MethodError primarycensored(Normal(1,2), 7)
end

@testitem "Test constructor" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
    @test typeof(use_dist) <: PrimaryCensored.PrimaryCensoredDist
end

@testitem "Test random generation" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
    @test length(rand(use_dist,10)) == 10
end

@testitem  "Test cdf method" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
    @test cdf(use_dist,1e8) ≈ 1.0
end


@testitem "Test ccdf" begin
    using Distributions
    use_dist = primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
    @test ccdf(use_dist,1e8) ≈ 0.0
    @test ccdf(use_dist, 0.0) ≈ 1
end

