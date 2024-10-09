@testitem "Testing censored_pmf function" begin
    using Distributions
    unif_dist = Uniform(0, 1)
    main_example_dist = primarycensored(Exponential(1.0), unif_dist)

    # Test case 1: Testing with a non-negative distribution
    @testset "Test case 1" begin
        dist = primarycensored(Normal(), unif_dist)
        @test_throws AssertionError censored_pmf(dist, Δd = 1.0, D = 3.0)
    end

    # Test case 2: Testing with Δd = 0.0
    @testset "Test case 2" begin
        @test_throws AssertionError censored_pmf(main_example_dist, Δd = 0.0, D = 3.0)
    end

    @testset "Test case 3" begin
        @test_throws AssertionError censored_pmf(main_example_dist, Δd = 3.0, D = 1.0)
    end

    # Test case 4: Testing output against expected PMF basic version - double
    # interval censoring
    @testset "Test case 4: censored pmf" begin
        expected_pmf_uncond = [exp(-1)
                               [(1 - exp(-1)) * (exp(1) - 1) * exp(-s) for s in 1:9]]
        pmf = censored_pmf(main_example_dist; Δd = 1.0, D = 10.0)
        @test expected_pmf_uncond≈pmf atol=1e-15
    end

    @testset "Test case 5: D not a multiple of Δd" begin
        @test_throws AssertionError censored_pmf(main_example_dist, Δd = 1.0, D = 3.5)
    end

    @testset "Test case 6: testing default choice of D" begin
        pmf = censored_pmf(main_example_dist, Δd = 1.0)
        #Check the normalisation constant is > 0.99 for analytical solution
        expected_pmf_uncond = [exp(-1)
                               [(1 - exp(-1)) * (exp(1) - 1) * exp(-s)
                                for s in 1:length(pmf)]]
        @test sum(expected_pmf_uncond) > 0.99
    end

    @testset "Check CDF function" begin
        expected_pmf_uncond = [exp(-1)
                               [(1 - exp(-1)) * (exp(1) - 1) * exp(-s) for s in 1:9]]
        expected_cdf = [0.0; cumsum(expected_pmf_uncond)]
        calc_cdf = censored_cdf(main_example_dist; Δd = 1.0, D = 10.0)
        @test expected_cdf≈calc_cdf atol=1e-15
    end
end
