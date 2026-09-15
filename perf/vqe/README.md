# Variational Quantum Eigensolver (VQE)

A simple VQE training through exact tensor network contraction and backpropagation.
Because it uses exact tensor network contraction, it cannot scale to large number of layers but it should be able to scale to mid-range number of qubits (around 50) and shallow circuits.

It uses a "Efficient SU(2)" circuit ansatz for the VQE.

## Setup

Tenet.jl and Tangles.jl are not registered in the General registry of packages.
You need to add the Quantic registry of packages for Julia to find them:

```julia
]registry add https://github.com/bsc-quantic/Registry.git
```

> [!WARNING]
> Reactant.jl version is fixed to 0.2.138 due to a bug with the rev diff rule of `stablehlo.dynamic_update_slice`
