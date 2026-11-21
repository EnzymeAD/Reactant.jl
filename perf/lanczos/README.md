# Lanczos benchmark

This benchmark implements the Lanczos algorithm for top-k eigenvalue decomposition of a Hermitian matrix.

> **require** $A, b$
>
> $\beta_1 v_1 = b$ s.t. $||v_1|| = 1$
>
> **for** $k = 1, 2, \dots$
>
> $\quad\alpha_k = v_k^\dagger A v_k$
>
> $\quad\beta_{k+1} v_{k+1} = A v_k - \alpha_k v_k - \beta_k^* v_{k-1}$
>
> **end**

$A V_k = V_k T_k + \beta_{k+1,k} v_{k+1}e_k^T = V_{k+1}T_{k+1,k}$

$V_k^\dagger V_k = \mathbb{I}_k$
