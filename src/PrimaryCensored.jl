module PrimaryCensored

# Non-submodule imports
using DocStringExtensions
using Distributions
using Random
using Integrals

# Exported constructors
export primarycensored, discretise, discretize, within_interval_censored

include("docstrings.jl")

include("PrimaryCensoredDist.jl")
include("DiscretisedDist.jl")
include("WithinIntervalCensored.jl")

end
