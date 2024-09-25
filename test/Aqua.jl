
@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(
        primarycensored, ambiguities = false, persistent_tasks = false
    )
    Aqua.test_ambiguities(primarycensored)
end
