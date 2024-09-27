using Random
using Test
using Distributions
using PrimaryCensored

function generate_object()
    primarycensored(LogNormal(3.5, 1.5), Uniform(1,2))
end

# Bad inputs
@test_throws MethodError primarycensored(Normal(1,2), 7)

# Test constructor
@test typeof(generate_object()) <: PrimaryCensored.PrimaryCensoredDist

# Test random generation
@test length(rand(generate_object(),10)) == 10

# Test cdf method
@test cdf(generate_object(),1e8) ≈ 1.0

# Test ccdf
@test ccdf(generate_object(),1e8) ≈ 0.0
@test ccdf(generate_object(), 0.0) ≈ 1