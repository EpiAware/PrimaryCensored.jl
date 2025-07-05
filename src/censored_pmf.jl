"""
Internal function to check censored_pmf arguments and return the time steps of the rightmost limits of the censor intervals.
"""
function _check_and_give_ts(dist::primarycensored, Δd, D, upper)
    @assert minimum(dist.uncensored)>=0.0 "Delay distribution must be non-negative."
    @assert Δd>0.0 "Δd must be positive."

    if isnothing(D)
        x_upr = invlogcdf(dist.uncensored, log(upper))
        D = round(Int64, x_upr / Δd) * Δd
    end

    @assert D>=Δd "D can't be shorter than Δd."

    ts = Δd:Δd:D |> collect

    @assert ts[end]==D "D must be a multiple of Δd."

    return ts
end

@doc raw"
Create a discrete probability cumulative distribution function (CDF) from a given `primarycensored`,
specialized for the case where all observsations are censored into disjoint intervals of width `Δd`. A
canonical example of this would be daily or weekly reporting.

NB: `censored_cdf` returns the _non-truncated_ CDF, i.e. the CDF without conditioning on the
secondary event occuring either before or after some time. Truncation can be applied by conditioning.


# Arguments
- `dist`: The `primarycensored` distribution from which to evaluate the CDF.
- `Δd`: The step size for discretizing the domain. Default is 1.0.
- `D`: The upper bound of the domain. Must be greater than `Δd`. Default `D = nothing`
indicates that the coverage of the distribution returned at its `upper`th percentile, _if the delay started at the
beginning of its primary interval rounded to nearest multiple of `Δd`.


# Returns
- A vector representing the CDF with 0.0 appended at the beginning.

# Raises
- `AssertionError` if the minimum value of `dist` is negative.
- `AssertionError` if `Δd` is not positive.
- `AssertionError` if `D` is shorter than `Δd`.
- `AssertionError` if `D` is not a multiple of `Δd`.

# Examples

```jldoctest filter
using PrimaryCensored, Distributions

dist = primarycensored(Exponential(1.0), Uniform(0, 1))

censored_cdf(dist; D = 10) |>
    p -> round.(p, digits=3)

# output
11-element Vector{Float64}:
 0.0
 0.368
 0.767
 0.914
 0.969
 0.988
 0.996
 0.998
 0.999
 1.0
 1.0
```
"
function censored_cdf(dist::primarycensored; Δd = 1.0, D = nothing, upper = 0.99)
    ts = _check_and_give_ts(dist, Δd, D, upper)
    cens_F = ts .|> t -> cdf(dist, t)
    return [0.0; cens_F]
end

@doc raw"
Create a discrete probability probability mass function (PMF) from a given `primarycensored`,
specialized for the case where all observsations are censored into disjoint intervals of width `Δd`. A
canonical example of this would be daily or weekly reporting.

NB: `censored_pmf` returns the _non-truncated_ PMF, i.e. the PMF without conditioning on the
secondary event occuring either before or after some time. Truncation can be applied by conditioning.


# Arguments
- `dist`: The `primarycensored` distribution from which to evaluate the CDF.
- `Δd`: The step size for discretizing the domain. Default is 1.0.
- `D`: The upper bound of the domain. Must be greater than `Δd`. Default `D = nothing`
indicates that the coverage of the distribution returned at its `upper`th percentile, _if the delay started at the
beginning of its primary interval rounded to nearest multiple of `Δd`.


# Returns
- A vector representing the CDF with 0.0 appended at the beginning.

# Raises
- `AssertionError` if the minimum value of `dist` is negative.
- `AssertionError` if `Δd` is not positive.
- `AssertionError` if `D` is shorter than `Δd`.
- `AssertionError` if `D` is not a multiple of `Δd`.

# Examples

```jldoctest filter
using PrimaryCensored, Distributions

dist = primarycensored(Exponential(1.0), Uniform(0, 1))

censored_pmf(dist; D = 10) |>
    p -> round.(p, digits=3)

# output
10-element Vector{Float64}:
 0.368
 0.4
 0.147
 0.054
 0.02
 0.007
 0.003
 0.001
 0.0
 0.0
```
"
function censored_pmf(dist::primarycensored; Δd = 1.0, D = nothing, upper = 0.99)
    cens_cdf = censored_cdf(dist; Δd, D, upper)
    return cens_cdf |> diff
end
