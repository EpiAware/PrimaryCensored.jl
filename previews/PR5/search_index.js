var documenterSearchIndex = {"docs":
[{"location":"getting-started/installation/#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"getting-started/installation/","page":"Installation","title":"Installation","text":"Eventually, primarycensored is likely to be added to the Julia registry. Until then, you can install it from this repository by running the following command in the Julia REPL:","category":"page"},{"location":"getting-started/installation/","page":"Installation","title":"Installation","text":"using Pkg; Pkg.add(url=\"https://github.com/epinowcast/primarycensored.jl\")","category":"page"},{"location":"lib/internals/#Internal-Documentation","page":"Internal API","title":"Internal Documentation","text":"","category":"section"},{"location":"lib/internals/","page":"Internal API","title":"Internal API","text":"Documentation for primarycensored.jl's internal interface.","category":"page"},{"location":"lib/internals/#Contents","page":"Internal API","title":"Contents","text":"","category":"section"},{"location":"lib/internals/","page":"Internal API","title":"Internal API","text":"Pages = [\"internals.md\"]\nDepth = 2:2","category":"page"},{"location":"lib/internals/#Index","page":"Internal API","title":"Index","text":"","category":"section"},{"location":"lib/internals/","page":"Internal API","title":"Internal API","text":"Pages = [\"internals.md\"]","category":"page"},{"location":"lib/internals/#Internal-API","page":"Internal API","title":"Internal API","text":"","category":"section"},{"location":"lib/internals/","page":"Internal API","title":"Internal API","text":"Modules = [primarycensored]\nPublic = false","category":"page"},{"location":"lib/public/#Public-Documentation","page":"Public API","title":"Public Documentation","text":"","category":"section"},{"location":"lib/public/","page":"Public API","title":"Public API","text":"Documentation for primarycensored.jl's public interface.","category":"page"},{"location":"lib/public/","page":"Public API","title":"Public API","text":"See the Internals section of the manual for internal package docs covering all submodules.","category":"page"},{"location":"lib/public/#Contents","page":"Public API","title":"Contents","text":"","category":"section"},{"location":"lib/public/","page":"Public API","title":"Public API","text":"Pages = [\"public.md\"]\nDepth = 2:2","category":"page"},{"location":"lib/public/#Index","page":"Public API","title":"Index","text":"","category":"section"},{"location":"lib/public/","page":"Public API","title":"Public API","text":"Pages = [\"public.md\"]","category":"page"},{"location":"lib/public/#Public-API","page":"Public API","title":"Public API","text":"","category":"section"},{"location":"lib/public/","page":"Public API","title":"Public API","text":"Modules = [primarycensored]\nPrivate = false","category":"page"},{"location":"developer/#developer","page":"Overview","title":"Developer documentation","text":"","category":"section"},{"location":"developer/","page":"Overview","title":"Overview","text":"Welcome to the primarycensored developer documentation! This section is designed to help you get started with developing the package.","category":"page"},{"location":"getting-started/quickstart/#Quickstart","page":"Quickstart","title":"Quickstart","text":"","category":"section"},{"location":"getting-started/quickstart/","page":"Quickstart","title":"Quickstart","text":"Get up and running with primarycensored in just a few minutes using this quickstart guide.","category":"page"},{"location":"release-notes/","page":"Release notes","title":"Release notes","text":"EditURL = \"https://github.com/JuliaDocs/Documenter.jl/blob/master/CHANGELOG.md\"","category":"page"},{"location":"release-notes/#Release-notes","page":"Release notes","title":"Release notes","text":"","category":"section"},{"location":"release-notes/","page":"Release notes","title":"Release notes","text":"The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.","category":"page"},{"location":"release-notes/#Unreleased","page":"Release notes","title":"Unreleased","text":"","category":"section"},{"location":"release-notes/#Added","page":"Release notes","title":"Added","text":"","category":"section"},{"location":"release-notes/#Changed","page":"Release notes","title":"Changed","text":"","category":"section"},{"location":"release-notes/#Fixed","page":"Release notes","title":"Fixed","text":"","category":"section"},{"location":"getting-started/explainers/#Explainers","page":"Overview","title":"Explainers","text":"","category":"section"},{"location":"getting-started/explainers/","page":"Overview","title":"Overview","text":"This section contains a series of explainers that provide a detailed overview of the primarycensored platform and its features. These explainers are designed to help you understand the platform and its capabilities, and to provide you with the information you need to get started using primarycensored. See the sidebar for the list of explainers.","category":"page"},{"location":"getting-started/#getting-started","page":"Overview","title":"Getting started","text":"","category":"section"},{"location":"getting-started/","page":"Overview","title":"Overview","text":"Note that this section of the documentation is still under construction. Please see replications for the most up-to-date information. Please feel free to contribute to the documentation by submitting a pull request.","category":"page"},{"location":"getting-started/","page":"Overview","title":"Overview","text":"Welcome to the primarycensored documentation! This section is designed to help you get started with the package. It includes a frequently asked questions (FAQ) section, a series of explainers that provide a detailed overview of the platform and its features, and tutorials that will help you get started with primarycensored for specific tasks. See the sidebar for the list of topics.","category":"page"},{"location":"lib/#api-reference","page":"Overview","title":"API reference","text":"","category":"section"},{"location":"lib/","page":"Overview","title":"Overview","text":"Welcome to the primarycensored API reference! This section is designed to help you understand the API of the package which is split into submodules.","category":"page"},{"location":"lib/","page":"Overview","title":"Overview","text":"The primarycensored package itself contains no functions or types. Instead, it re-exports the functions and types from its submodules. See the sidebar for the list of submodules.","category":"page"},{"location":"showcase/#showcase","page":"Overview","title":"primarycensored Showcase","text":"","category":"section"},{"location":"showcase/","page":"Overview","title":"Overview","text":"Here we showcase the capabilities of primarycensored in action. If you have a showcase you would like to add, please submit a pull request.","category":"page"},{"location":"developer/contributing/#Contributing","page":"Contributing","title":"Contributing","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"This page details the some of the guidelines that should be followed when contributing to this package. It is adapted from Documenter.jl.","category":"page"},{"location":"developer/contributing/#Branches","page":"Contributing","title":"Branches","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"release-* branches are used for tagged minor versions of this package. This follows the same approach used in the main Julia repository, albeit on a much more modest scale.","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Please open pull requests against the master branch rather than any of the release-* branches whenever possible.","category":"page"},{"location":"developer/contributing/#Backports","page":"Contributing","title":"Backports","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Bug fixes are backported to the release-* branches using git cherry-pick -x by a EpiAware member and will become available in point releases of that particular minor version of the package.","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Feel free to nominate commits that should be backported by opening an issue. Requests for new point releases to be tagged in METADATA.jl can also be made in the same way.","category":"page"},{"location":"developer/contributing/#release-*-branches","page":"Contributing","title":"release-* branches","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Each new minor version x.y.0 gets a branch called release-x.y (a protected branch).\nNew versions are usually tagged only from the release-x.y branches.\nFor patch releases, changes get backported to the release-x.y branch via a single PR with the standard name \"Backports for x.y.z\" and label \"Type: Backport\". The PR message links to all the PRs that are providing commits to the backport. The PR gets merged as a merge commit (i.e. not squashed).\nThe old release-* branches may be removed once they have outlived their usefulness.\nPatch version milestones are used to keep track of which PRs get backported etc.","category":"page"},{"location":"developer/contributing/#Style-Guide","page":"Contributing","title":"Style Guide","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Follow the style of the surrounding text when making changes. When adding new features please try to stick to the following points whenever applicable. This project follows the SciML style guide.","category":"page"},{"location":"developer/contributing/#Tests","page":"Contributing","title":"Tests","text":"","category":"section"},{"location":"developer/contributing/#Unit-tests","page":"Contributing","title":"Unit tests","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"As is conventional for Julia packages, unit tests are located at test/*.jl with the entrypoint test/runtests.jl.","category":"page"},{"location":"developer/contributing/#End-to-end-testing","page":"Contributing","title":"End to end testing","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Tests that build example package docs from source and inspect the results (end to end tests) are located in /test/examples. The main entry points are test/examples/make.jl for building and test/examples/test.jl for doing some basic checks on the generated outputs.","category":"page"},{"location":"developer/contributing/#Pluto-usage-in-showcase-documentation","page":"Contributing","title":"Pluto usage in showcase documentation","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Some of the showcase examples in primarycensored/docs/src/showcase use Pluto.jl notebooks for the underlying computation. The output of the notebooks is rendered into HTML for inclusion in the documentation in two steps:","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"PlutoStaticHTML.jl converts the notebook with output into a machine-readable .md format.\nDocumenter.jl renders the .md file into HTML for inclusion in the documentation during the build process.","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"For other examples of using Pluto to generate documentation see the examples shown here.","category":"page"},{"location":"developer/contributing/#Running-Pluto-notebooks-from-primarycensored-locally","page":"Contributing","title":"Running Pluto notebooks from primarycensored locally","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"To run the Pluto.jl scripts in the primarycensored documentation directly from the source code you can do these steps:","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Install Pluto.jl locally. We recommend using the version of Pluto that is pinned in the Project.toml file defining the documentation environment.\nClone the primarycensored repository.\nStart Pluto.jl either from REPL (see the Pluto.jl documentation) or from the command line with the shell script primarycensored/docs/pluto-scripts.sh.\nFrom the Pluto.jl interface, navigate to the Pluto.jl script you want to run.","category":"page"},{"location":"developer/contributing/#Contributing-to-Pluto-notebooks-in-primarycensored-documentation","page":"Contributing","title":"Contributing to Pluto notebooks in primarycensored documentation","text":"","category":"section"},{"location":"developer/contributing/#Modifying-an-existing-Pluto-notebook","page":"Contributing","title":"Modifying an existing Pluto notebook","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Committing changes to the Pluto.jl notebooks in the EpiAware documentation is the same as committing changes to any other part of the repository. However, please note that we expect the following features for the environment management of the notebooks:","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Use the environment determined by the Project.toml file in the primarycensored/docs directory. If you want extra packages, add them to this environment.\nUse the version of primarycensored that is used in these notebooks to be the version of primarycensored on the branch being pull requested into main. To do this use the Pkg.develop function.","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"To do this you can use the following code snippet in the Pluto notebook:","category":"page"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"# Determine the relative path to the `EpiAware/docs` directory\ndocs_dir = dirname(dirname(dirname(dirname(@__DIR__))))\n# Determine the relative path to the `EpiAware` package directory\npkg_dir = dirname(docs_dir)\n\nusing Pkg: Pkg\nPkg.activate(docs_dir)\nPkg.develop(; path = pkg_dir)\nPkg.instantiate()","category":"page"},{"location":"developer/contributing/#Adding-a-new-Pluto-notebook","page":"Contributing","title":"Adding a new Pluto notebook","text":"","category":"section"},{"location":"developer/contributing/","page":"Contributing","title":"Contributing","text":"Adding a new Pluto.jl notebook to the primarycensored documentation is the same as adding any other file to the repository. However, in addition to following the guidelines for modifying an existing notebook, please note that the new notebook is added to the set of notebook builds using build in the primarycensored/docs/make.jl file. This will generate an .md of the same name as the notebook which can be rendered when makedocs is run. For this document to be added to the overall documentation the path to the .md file must be added to the Pages array defined in primarycensored/docs/pages.jl.","category":"page"},{"location":"developer/checklist/#Checklists","page":"Release checklist","title":"Checklists","text":"","category":"section"},{"location":"developer/checklist/","page":"Release checklist","title":"Release checklist","text":"The purpose of this page is to collate a series of checklists for commonly performed changes to the source code of EpiAware. It has been adapted from Documenter.jl.","category":"page"},{"location":"developer/checklist/","page":"Release checklist","title":"Release checklist","text":"In each case, copy the checklist into the description of the pull request.","category":"page"},{"location":"developer/checklist/#Making-a-release","page":"Release checklist","title":"Making a release","text":"","category":"section"},{"location":"developer/checklist/","page":"Release checklist","title":"Release checklist","text":"In preparation for a release, use the following checklist. These steps should be performed on a branch with an open pull request, either for a topic branch, or for a new branch release-1.y.z (\"Release version 1.y.z\") if multiple changes have accumulated on the master branch since the last release.","category":"page"},{"location":"developer/checklist/","page":"Release checklist","title":"Release checklist","text":"## Pre-release\n\n - [ ] Change the version number in `Project.toml`\n   * If the release is breaking, increment MAJOR\n   * If the release adds a new user-visible feature, increment MINOR\n   * Otherwise (bug-fixes, documentation improvements), increment PATCH\n - [ ] Update `CHANGELOG.md`, following the existing style (in particular, make sure that the change log for this version has the correct version number and date).\n - [ ] Run `make changelog`, to make sure that all the issue references in `CHANGELOG.md` are up to date.\n - [ ] Check that the commit messages in this PR do not contain `[ci skip]`\n - [ ] Run https://github.com/JuliaDocs/Documenter.jl/actions/workflows/regression-tests.yml\n       using a `workflow_dispatch` trigger to check for any changes that broke extensions.\n\n## The release\n\n - [ ] After merging the pull request, tag the release. There are two options for this:\n\n   1. [Comment `[at]JuliaRegistrator register` on the GitHub commit.](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app)\n   2. Use [JuliaHub's package registration feature](https://help.juliahub.com/juliahub/stable/contribute/#registrator) to trigger the registration.\n\n   Either of those should automatically publish a new version to the Julia registry.\n - Once registered, the `TagBot.yml` workflow should create a tag, and rebuild the documentation for this tag.\n - These steps can take quite a bit of time (1 hour or more), so don't be surprised if the new documentation takes a while to appear.","category":"page"},{"location":"overview/#overview","page":"Overview","title":"Overview of the primarycensored","text":"","category":"section"},{"location":"getting-started/tutorials/#Tutorials","page":"Overview","title":"Tutorials","text":"","category":"section"},{"location":"getting-started/tutorials/","page":"Overview","title":"Overview","text":"This section contains tutorials that will help you get started with primarycensored for specific tasks. See the sidebar for the list of tutorials.","category":"page"},{"location":"getting-started/faq/#Frequently-asked-questions","page":"Frequently asked questions","title":"Frequently asked questions","text":"","category":"section"},{"location":"getting-started/faq/","page":"Frequently asked questions","title":"Frequently asked questions","text":"This page contains a list of frequently asked questions about the primarycensored package. If you have a question that is not answered here, please open a discussion on the GitHub repository.","category":"page"},{"location":"getting-started/faq/","page":"Frequently asked questions","title":"Frequently asked questions","text":"Pages = [\"lib/getting-started/faq.md\"]","category":"page"},{"location":"getting-started/faq/#Pluto-notebooks","page":"Frequently asked questions","title":"Pluto notebooks","text":"","category":"section"},{"location":"getting-started/faq/","page":"Frequently asked questions","title":"Frequently asked questions","text":"In some of the showcase examples in primarycensored/docs/src/showcase we use Pluto.jl notebooks for the underlying computation. As well as reading the code blocks and output of the notebooks in this documentation, you can also run these notebooks by cloning primarycensored and running the notebooks with Pluto.jl (for further details see developer notes).","category":"page"},{"location":"getting-started/faq/","page":"Frequently asked questions","title":"Frequently asked questions","text":"It should be noted that Pluto.jl notebooks are reactive, meaning that they re-run downstream code after changes with downstreaming determined by a tree of dependent code blocks. This is different from the standard Julia REPL, and some other notebook formats (e.g. .ipynb). In Pluto each code block is a single lines of code or encapsulated by let ... end and begin ... end. The difference between let ... end blocks and begin ... end blocks are that the let ... end type of code block only adds the final output/return value of the block to scope, like an anonymous function, whereas begin ... end executes each line and adds defined variables to scope.","category":"page"},{"location":"getting-started/faq/","page":"Frequently asked questions","title":"Frequently asked questions","text":"For installation instructions and more information and documentation on Pluto.jl see the Pluto.jl documentation.","category":"page"},{"location":"#primarycensored.jl","page":"primarycensored.jl: Primary event censored distributions","title":"primarycensored.jl","text":"","category":"section"},{"location":"","page":"primarycensored.jl: Primary event censored distributions","title":"primarycensored.jl: Primary event censored distributions","text":"Primary event censored distributions for Julia.","category":"page"},{"location":"#Where-to-start","page":"primarycensored.jl: Primary event censored distributions","title":"Where to start","text":"","category":"section"},{"location":"","page":"primarycensored.jl: Primary event censored distributions","title":"primarycensored.jl: Primary event censored distributions","text":"Want to get started running code? Check out the Getting Started Tutorials.\nWhat is primarycensored? Check out our Overview.\nWant to see some end-to-end examples? Check out our primarycensored showcase.\nWant to understand the API? Check out our API Reference.\nWant to chat with someone about primarycensored? Post on our GitHub Discussions.\nWant to contribute to primarycensored? Check out our Developer documentation.\nWant to see our code? Check out our GitHub Repository.","category":"page"}]
}
