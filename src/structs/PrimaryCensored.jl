@doc raw"
Generic wrapper for a [`censored`](@ref) distribution.

# Contructors

- `PrimaryCensored(; uncensored::D, censoring::D)`: Constructs a `PrimaryCensored` distribution with the specified uncensored and censoring distributions.
- `PrimaryCensored(uncensored::D, censoring::D)`: Constructs a `PrimaryCensored` distribution with the specified uncensored and censoring distributions.

# Examples

```@example
using primarycensored, Distributions

uncensored = Normal()
censoring = Uniform(0, 1)

d = PrimaryCensored(uncensored, censoring)

rand(d)
```
"
@kwdef struct PrimaryCensored{D1 <: UnivariateDistribution, D2 <: UnivariateDistribution} <:
              Distributions.UnivariateDistribution{Distributions.ValueSupport}
    "The original (uncensored) distribution."
    uncensored::D1
    "The primary event censoring distribution."
    censoring::D2
end

function params(d::PrimaryCensored)
    d0params = params(d.uncensored)
    d1params = params(d.censoring)
    return (d0params..., d1params...)
end

Base.eltype(::Type{<:PrimaryCensored{D}}) where {D} = promote_type(eltype(D), eltype(D))

function Distributions.cdf(d::PrimaryCensored, x::Real)
    result = cdf(d.uncensored, x)
    return (result)
end

function Distributions.logcdf(d::PrimaryCensored, x::Real)
    result = logcdf(d.uncensored, x)
    return result
end

function Distributions.ccdf(d::PrimaryCensored, x::Real)
    result = ccdf(d.uncensored, x)
    return result
end

function Distributions.logccdf(d::PrimaryCensored, x::Real)
    result = logccdf(d.uncensored, x)
    return result
end

#### Sampling

function Base.rand(rng::AbstractRNG, d::PrimaryCensored)
    rand(rng, d.uncensored) + rand(rng, d.censoring)
end
