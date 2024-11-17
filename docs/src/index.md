```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Reactant Docs
  text: TODO
  tagline: TODO
  actions:
    - theme: brand
      text: Tutorials
      link: /tutorials
    - theme: alt
      text: API Reference 📚
      link: /api/api
    - theme: alt
      text: View on GitHub
      link: https://github.com/EnzymeAD/Reactant.jl
  image:
    src: /logo.svg
    alt: Reactant.jl

features:
  - icon: 🚀
    title: TODO
    details: TODO
    link: TODO

  - icon: 🧑‍🔬
    title: TODO
    details: TODO
    link: TODO

  - icon: 🧩
    title: TODO
    details: TODO
    link: TODO

  - icon: 🧪
    title: TODO
    details: TODO
    link: TODO
---
```

## How to Install Reactant.jl?

Its easy to install Reactant.jl. Since Reactant.jl is registered in the Julia General
registry, you can simply run the following command in the Julia REPL:

```julia
julia> using Pkg
julia> Pkg.add("Reactant")
```

If you want to use the latest unreleased version of Reactant.jl, you can run the following
command:

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/EnzymeAD/Reactant.jl")
```

## Select an Accelerator Backend

:::code-group

```julia [CPUs]
using Reactant
Reactant.set_default_backend("cpu")
```

```julia [NVIDIA GPUs]
using Reactant
Reactant.set_default_backend("gpu")
```

```julia [Cloud TPUs]
using Reactant
Reactant.set_default_backend("tpu")
```

:::
