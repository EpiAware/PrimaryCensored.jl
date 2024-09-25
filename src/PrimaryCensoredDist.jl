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

@doc raw"
Generic wrapper for a [`censored`](@ref) distribution.

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

function primarycensored(
        uncensored::UnivariateDistribution, censoring::UnivariateDistribution)
    return PrimaryCensoredDist(uncensored, censoring)
end

function params(d::PrimaryCensoredDist)
    d0params = params(d.uncensored)
    d1params = params(d.censoring)
    return (d0params..., d1params...)
end

Base.eltype(::Type{<:PrimaryCensoredDist{D}}) where {D} = promote_type(eltype(D), eltype(D))

function Distributions.cdf(d::PrimaryCensoredDist, x::Real)
    result = cdf(d.uncensored, x)
    return (result)
end

function Distributions.logcdf(d::PrimaryCensoredDist, x::Real)
    result = logcdf(d.uncensored, x)
    return result
end

function Distributions.ccdf(d::PrimaryCensoredDist, x::Real)
    result = ccdf(d.uncensored, x)
    return result
end

function Distributions.logccdf(d::PrimaryCensoredDist, x::Real)
    result = logccdf(d.uncensored, x)
    return result
end

#### Sampling

function Base.rand(rng::AbstractRNG, d::PrimaryCensoredDist)
    rand(rng, d.uncensored) + rand(rng, d.censoring)
end
