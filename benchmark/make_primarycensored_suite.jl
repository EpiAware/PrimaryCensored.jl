"""
   A custom wrapper for the `TuringBenchmarking.make_turing_suite` that adds primarycensored specific defaults.
"""
function make_primarycensored_suite(model; check = true,
        adbackends = [:forwarddiff, :reversediff, :reversediff_compiled])
    suite = TuringBenchmarking.make_turing_suite(
        model; check = check, adbackends = adbackends)
    return suite
end
