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
- WithinIntervalCensor object

"
#! TODO Struct
struct WithinIntervalCensor

end


#! Method of evaluation
#! Gives full discretized interval censored pmf for some distribution
#! Note the -1 to represent indexing from zero

function within_interval_censored(dist, obs_time, swindow::Real)
    if obs_time < 0
        throw(error("`obs_time` = $obs_time and should not be negative"))
    end

    out = zeros(obs_time)
    for n in eachindex(out)
        out[n] = cdf(dist, n + swindow - 1) - cdf(dist, n - 1)
    end
    # Normalise
    out = out ./ sum(out)

    return out
end




