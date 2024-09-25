
@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(
        PrimaryCensored, ambiguities = false, persistent_tasks = false
    )
    Aqua.test_ambiguities(PrimaryCensored)
end
