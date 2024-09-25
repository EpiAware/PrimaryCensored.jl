using Distributions
using Random

function rand_primarycensored(
        rng::AbstractRNG, delay::UnivariateDistribution, primary::UnivariateDistribution)
    p = rand(rng, primary)
    d = rand(rng, delay)
    total_delay = p + d
    return total_delay
end

rand_primarycensored(Random.default_rng(), LogNormal(), Uniform())
