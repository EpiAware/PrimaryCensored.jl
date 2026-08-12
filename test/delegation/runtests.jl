#!/usr/bin/env julia
# Delegation harness: the pre-migration implementation
# (`test/fixtures/LegacyPrimaryCensored`) run side by side with the
# `ConvolvedDistributions`-backed one, against an independent BigFloat
# reference. Own project because the fixture is a path source, which
# Julia 1.10 cannot resolve.
#
#   julia --project=test/delegation test/delegation/runtests.jl
#
# `run_tests(@__DIR__)` confines discovery to this directory, so the
# in-repo `worktrees/` siblings are never scanned.

using TestItemRunner

TestItemRunner.run_tests(@__DIR__)
