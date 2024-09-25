using Documenter
using primarycensored
using Pluto: Configuration.CompilerOptions
using PlutoStaticHTML

include("changelog.jl")
include("pages.jl")
include("build.jl")

build("getting-started")
build("getting-started/tutorials")

DocMeta.setdocmeta!(
    primarycensored, :DocTestSetup, :(using primarycensored); recursive = true)

makedocs(; sitename = "primarycensored.jl",
    authors = "Samuel Brand, Sam Abbott, and contributors",
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs, :linkcheck, :autodocs_block],
    modules = [primarycensored],
    pages = pages,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax3(),
        size_threshold = 6000 * 2^10,
        size_threshold_warn = 2000 * 2^10
    )
)

deploydocs(
    repo = "github.com/epinowcast/primarycensored.jl.git",
    target = "build",
    push_preview = true
)
