# Timeseries convolution with continuous and time-varying delays

md"""
This tutorial shows what `convolve_series` does by plotting it. It convolves a
numeric `series` with a delay to produce an expected downstream count curve —
the renewal-style observation layer that turns an infection curve into an
expected counts / reports curve.

`convolve_series` lives in ConvolvedDistributions.jl; loading CensoredDistributions
alongside it activates a bridge that understands our censored delays. In
particular it lets you convolve a **raw continuous delay** (discretised for you
via `double_interval_censored`), a pre-built **interval-censored delay**, and a
**time-varying** sequence of delays.
"""

md"""
### What are we going to do

1. Build a delay and a synthetic infection series.
2. Convolve the infection series into an expected count curve:
   - with a raw continuous delay,
   - with a weekly (interval-censored) delay,
   - with a time-varying delay.
"""

md"""
### What might I need to know before starting

`convolve_series(delay, series)` is a discrete convolution. The delay is
turned into a probability mass function on a lag grid, and the series entry at
time `i` is smeared forward by that mass. For a continuous delay the grid step
is the `interval` keyword (default `1`), so be sure the series and the delay are
on the same time unit.
"""

md"""
### Packages used
"""

using CensoredDistributions, ConvolvedDistributions, Distributions
using CairoMakie, AlgebraOfGraphics, DataFramesMeta

CairoMakie.activate!(type = "png", px_per_unit = 2)

md"""
### A raw continuous delay

Pass a continuous distribution straight in. It is discretised with
`double_interval_censored` on a unit grid by default, so `series` entry `i` is
read at time `(i - 1)`.
"""

t = 0:40
infections = 100 .* exp.(-((t .- 12.0) .^ 2) ./ 30.0)
expected = convolve_series(LogNormal(1.5, 0.75), infections)

timeseries_df = vcat(
    DataFrame(t = t, count = infections, series = "Infections"),
    DataFrame(t = t, count = expected, series = "Expected reports")
)
draw(
    data(timeseries_df) * mapping(:t, :count, color = :series) *
    visual(Lines, linewidth = 2);
    axis = (xlabel = "Day", ylabel = "Expected count")
)

md"""
### A weekly (interval-censored) delay

The `interval` keyword sets the delay's grid step, and any other keyword
passes through to [`double_interval_censored`](@ref) (e.g. a `primary_event`).
Setting `interval = 7` reads the delay on a weekly grid, so `series` must
also be on a weekly grid: entry `i` is then read at week `(i - 1)`, not day
`(i - 1)`. Aggregate the daily infections into weekly totals first.

You can reach that weekly grid two ways: pass the raw distribution with
`interval = 7`, or discretise it yourself with `double_interval_censored`
(e.g. to control the binning or the primary event) and pass the result
directly. Both give the same PMF.
"""

weekly_infections = [sum(infections[(7i + 1):min(7i + 7, length(infections))])
                     for i in 0:5]

expected_weekly = convolve_series(
    LogNormal(1.5, 0.75), weekly_infections; interval = 7)

delay_weekly = double_interval_censored(LogNormal(1.5, 0.75); interval = 7)
expected_weekly_built = convolve_series(delay_weekly, weekly_infections);

expected_weekly ≈ expected_weekly_built

md"""
### A time-varying delay

`convolve_series` also accepts a **vector** of delays, one per series entry.
Each entry is smeared forward through its own delay, which lets the delay
distribution change over time (e.g. reporting delays that shorten as an
outbreak matures). As with a single delay, each entry can be a raw
continuous distribution, discretised for you on the unit grid.
"""

delays = [LogNormal(m, 0.6)
          for m in range(1.0, 1.8; length = length(t))]
expected_timevarying = convolve_series(delays, infections)

timeseries_timevarying_df = vcat(
    DataFrame(t = t, count = infections, series = "Infections"),
    DataFrame(t = t, count = expected_timevarying,
        series = "Expected reports (time-varying)")
)
draw(
    data(timeseries_timevarying_df) * mapping(:t, :count, color = :series) *
    visual(Lines, linewidth = 2);
    axis = (xlabel = "Day", ylabel = "Expected count")
)

md"""
### Summary

- `convolve_series(delay, series)` turns an infection series into an expected
  counts curve = a convolution with the delay's PMF.
- A raw continuous delay is discretised for you (`interval` keyword, other
  kwargs to `double_interval_censored`); a bare `primary_censored` delay needs
  an explicit secondary interval censoring step.
- A pre-built `interval_censored` delay is read on its own grid.
- A vector of delays gives a time-varying convolution.
"""
