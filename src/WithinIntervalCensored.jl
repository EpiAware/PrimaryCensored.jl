@doc raw"
Implement a censoring function. 
Takes as and input the the object from truncated form of the 
primarycensored object. This object only needs to have a method
for the cumulative distribution function.

# Arguments
- `dist` the truncated primarycensored distribution
- `obs_time` the observation time of interest
- `swindow` the censoring window for the interval
- `pwindow` primary censoring window

# Returns
- WithinIntervalCensoredDist object

"
#! TODO Struct
struct WithinIntervalCensoredDist{D <: UnivariateDistribution, OBS <: Real, SWIN <:Real} <:
    Distributions.UnivariateDistribution{Distributions.ValueSupport}
    dist::D
    obs_time::OBS
    swindow::SWIN
end

@doc raw"
Generic wrapper for within window censored an object

# Contructors

"
function withinintervalcensoreddist(dist::UnivariateDistribution, obs_time::Real, swindow::Real)
    WithinIntervalCensoredDist(dist, obs_time, swindow)
end

#! Method of evaluation
#! Gives full discretized interval censored pmf for some distribution
#! Note the -1 to represent indexing from zero

@doc raw"
Returning a within interval censored pmf

# Arguments
- d an WithinIntervalCensoredDist object

# Returns
- discretized probability mass function

# Examples

```@example
    using PrimaryCensored, Distributions
    d= truncated(Normal(5,2),0,5)
    trunc_d = withinintervalcensoreddist(d, 4, 1.0)
    within_interval_censored(trunc_d)
```
"

function within_interval_censored(d::WithinIntervalCensoredDist)
    dobs = d.obs_time
    if d.obs_time < 0
        throw(error("`obs_time` = $dobs and should not be negative"))
    end

    obs_time_integer = floor(Int, dobs)

    out = zeros(obs_time_integer)
    for n in eachindex(out)
        out[n] = cdf(d.dist, n + d.swindow - 1) - cdf(d.dist, n - 1)
    end
    # Normalise
    out = out ./ sum(out)

    return out
end

    