```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Reactant.jl Docs
  text: Optimizing Julia Functions with MLIR
  tagline: Optimize Julia Functions With MLIR and XLA for High-Performance Execution on CPU, GPU, TPU and more.
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
    title: Fast & Device Agnostic
    details: Effortlessly execute your code on CPU, GPU, and TPU with MLIR and XLA.
    link: /introduction

  - icon: ∂
    title: Built-In MLIR AD
    details: Leverage Enzyme-Powered Automatic Differentiation to Differentiate MLIR Functions
    link: /introduction

  - icon: 🧩
    title: Composable
    details: Executes and optimizes generic Julia code without requiring special rewriting
    link: /introduction

  - icon: ⚡
    title: Compiler Optimizations
    details: Fancy MLIR Optimizations seamlessly optimize your Julia code
    link: /introduction
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
# Set backend to use a GPU if available
Reactant.set_default_backend("gpu")
# Set backend to specifically a CUDA GPU
# Reactant.set_default_backend("cuda")
```

```julia [AMD GPUs]
using Reactant
Reactant.set_default_backend("gpu")
# Set backend to specifically an AMD GPU
# Reactant.set_default_backend("rocm")
```

```julia [Cloud TPUs]
using Reactant
Reactant.set_default_backend("tpu")
```

```julia [Tenstorrent (Experimental)]
using Reactant
Reactant.set_default_backend("tt")
```
:::
