using Random
using Test
using Distributions
using PrimaryCensored

function generate_object()
    primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
end

use_dist = generate_object()

# Bad inputs
@testitem "Test struct type" begin
    @test_throws MethodError primarycensored(Normal(1,2), 7)
end

@testitem "Test constructor" begin
    @test typeof(use_dist) <: PrimaryCensored.PrimaryCensoredDist
end

@testitem "Test random generation" begin
    @test length(rand(use_dist,10)) == 10
end

@testitem  "Test cdf method" begin
    @test cdf(use_dist,1e8) ≈ 1.0
end


@testitem "Test ccdf" begin
    @test ccdf(use_dist,1e8) ≈ 0.0
    @test ccdf(use_dist, 0.0) ≈ 1
end

