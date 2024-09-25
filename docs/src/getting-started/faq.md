# Frequently asked questions

This page contains a list of frequently asked questions about the `PrimaryCensored` package. If you have a question that is not answered here, please open a discussion on the GitHub repository.

```@contents
Pages = ["lib/getting-started/faq.md"]
```

## Pluto notebooks

In some of the showcase examples in `PrimaryCensored/docs/src/showcase` we use [`Pluto.jl`](https://plutojl.org/) notebooks for the underlying computation. As well as reading the code blocks and output of the notebooks in this documentation, you can also run these notebooks by cloning `PrimaryCensored` and running the notebooks with `Pluto.jl` (for further details see [developer notes](@ref developer)).

It should be noted that `Pluto.jl` notebooks are reactive, meaning that they re-run downstream code after changes with downstreaming determined by a tree of dependent code blocks. This is different from the standard Julia REPL, and some other notebook formats (e.g. `.ipynb`). In `Pluto` each code block is a single lines of code or encapsulated by `let ... end` and `begin ... end`. The difference between `let ... end` blocks and `begin ... end` blocks are that the `let ... end` type of code block only adds the final output/return value of the block to scope, like an anonymous function, whereas `begin ... end` executes each line and adds defined variables to scope.

For installation instructions and more information and documentation on `Pluto.jl` see the [Pluto.jl documentation](https://plutojl.org/).
