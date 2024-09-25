@doc raw"
    Constructs a `PrimaryCensoredDist` distribution with the specified uncensored and censoring distributions.

    # Arguments
    - `uncensored::D`: The uncensored distribution.
    - `censoring::D`: The primary censoring distribution.

    # Returns
    - A `PrimaryCensored` distribution.

    # Examples
    ```@example
    using PrimaryCensored, Distributions

    uncensored = Normal()
    censoring = Uniform(0, 1)

    d = primarycensored(uncensored, censoring)
    ```
"
function primarycensored(
        uncensored::UnivariateDistribution, censoring::UnivariateDistribution)
    return PrimaryCensoredDist(uncensored, censoring)
end

@doc raw"
Generic wrapper for a primary event censored distribution.

# Contructors

- `PrimaryCensoredDist(; uncensored::D, censoring::D)`: Constructs a `PrimaryCensoredDist` distribution with the specified uncensored and censoring distributions.
# Examples

```@example
using PrimaryCensored, Distributions

uncensored = Normal()
censoring = Uniform(0, 1)

d = primarycensored(uncensored, censoring)

rand(d)
```
"
struct PrimaryCensoredDist{D1 <: UnivariateDistribution, D2 <: UnivariateDistribution} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
    "The original (uncensored) distribution."
    uncensored::D1
    "The primary event censoring distribution."
    censoring::D2
end

function params(d::PrimaryCensoredDist)
    d0params = params(d.uncensored)
    d1params = params(d.censoring)
    return (d0params..., d1params...)
end

Base.eltype(::Type{<:PrimaryCensoredDist{D}}) where {D} = promote_type(eltype(D), eltype(D))

function Distributions.cdf(d::PrimaryCensoredDist, x::Real)
    function f(u, x)
        return exp(logcdf(d.uncensored, x) - logpdf(d.censoring, x - u))
    end

    domain = (max(1e-6, x - maximum(d.censoring)), x)
    prob = IntegralProblem(f, domain, x)
    result = solve(prob, QuadGKJL())
    return (result)
end

function Distributions.logcdf(d::PrimaryCensoredDist, x::Real)
    result = log(cdf(d, x))
    return result
end

function Distributions.ccdf(d::PrimaryCensoredDist, x::Real)
    result = 1 - cdf(d, x)
    return result
end

function Distributions.logccdf(d::PrimaryCensoredDist, x::Real)
    result = log(ccdf(d, x))
    return result
end

#### Sampling

function Base.rand(rng::AbstractRNG, d::PrimaryCensoredDist)
    rand(rng, d.uncensored) + rand(rng, d.censoring)
end
