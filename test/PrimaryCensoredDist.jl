using Random
using Test
using Distributions
using PrimaryCensored

function generate_object()
    primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
end

# Bad inputs
@testitem "struct" begin
    @test_throws MethodError primarycensored(Normal(1,2), 7)
end


@testitem "Test constructor" begin
    @test typeof(generate_object()) <: PrimaryCensored.PrimaryCensoredDist
end


@testitem "Test random generation" begin
    @test length(rand(generate_object(),10)) == 10
end

@testitem  "Test cdf method" begin
    @test cdf(generate_object(),1e8) ≈ 1.0
end


@testitem "Test ccdf" begin
    @test ccdf(generate_object(),1e8) ≈ 0.0
    @test ccdf(generate_object(), 0.0) ≈ 1
end

