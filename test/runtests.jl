using TestItemRunner

# `:ad` items live under `test/ad/` and `:delegation` items under
# `test/delegation/`, each with its own project (the AD backends and
# the LegacyPrimaryCensored path fixture are not deps of the main test
# env) and run in their own dedicated CI. `@run_package_tests` walks
# `test/` recursively, so both are excluded by tag, not by directory.
# See `test/ad/runtests.jl` and `test/delegation/runtests.jl`.
_own_project(ti) = (:ad in ti.tags) || (:delegation in ti.tags)

# Filter tests based on command line arguments
if "skip_quality" in ARGS
    # Skip quality tests (JET, Aqua, formatting) used in CI for performance
    @run_package_tests filter = ti -> !(:quality in ti.tags) &&
                                      !_own_project(ti)
elseif "quality_only" in ARGS
    # Run only quality tests (Aqua, formatting, linting, doctests)
    @run_package_tests filter = ti -> :quality in ti.tags
elseif "readme_only" in ARGS
    # Run only README tests
    @run_package_tests filter = ti -> :readme in ti.tags
else
    # Run all tests (default for local development)
    @run_package_tests filter = ti -> !_own_project(ti)
end
