module Polybench
using Reactant

function kernel_correlation(D)
    m = size(D, 2)
    mean = zeros(eltype(D), m)::Reactant.TracedRArray{Float64,1}
    stddev = zeros(eltype(D), m)::Reactant.TracedRArray{Float64,1}
    corr = zeros(eltype(D), m, m)::Reactant.TracedRArray{Float64,2}
    for j in axes(D, 2)
        mean[j] = 0.0
        for i in axes(D, 1)
            mean[j] += D[i, j]
        end
        mean[j] /= size(D, 1)
    end

    for j in axes(D, 2)
        stddev[j] = 0.0
        for i in axes(D, 1)
            stddev[j] += (D[i, j] - mean[j]) * (D[i, j] - mean[j])
        end
        stddev[j] /= size(D, 1)
        stddev[j] = sqrt(stddev[j])
        stddev[j] = stddev[j] <= 0.1 ? 1.0 : stddev[j]
    end

    for i in axes(D, 1)
        for j in axes(D, 2)
            D[i, j] -= mean[j]
            D[i, j] /= sqrt(size(D, 1)) * stddev[j]
        end
    end

    for i in axes(D, 2)
        corr[i, i] = 1.0
        for j in (i + 1):m
            corr[i, j] = 0.0
            for k in axes(D, 1)
                corr[i, j] += D[k, i] * D[k, j]
            end
            corr[j, i] = corr[i, j]
        end
    end
    corr[m, m] = 1.0
    return corr
end

function kernel_covariance(D)
    m = size(D, 2)
    mean = zeros(eltype(D), m)
    cov = zeros(eltype(D), m, m)
    for j in axes(D, 2)
        mean[j] = 0.0
        for i in axes(D, 1)
            mean[j] += D[i, j]
        end
        mean[j] /= size(D, 1)
    end

    for i in axes(D, 1)
        for j in axes(D, 2)
            D[i, j] -= mean[j]
        end
    end

    for i in axes(D, 2)
        for j in axes(D, 2)
            cov[i, j] = 0.0
            for k in axes(D, 1)
                cov[i, j] += D[k, i] * D[k, j]
            end
            cov[i, j] /= size(D, 1) - 1.0
            cov[i, j] = cov[j, i]
        end
    end
    return cov
end

function kernel_gemm(alpha, beta, A, B)
    C = zeros(eltype(A), axes(A, 1), axes(B, 2))
    for i in axes(A, 1)
        for j in axes(B, 2)
            C[i, j] *= beta
        end
        for k in axes(A, 2)
            for j in axes(B, 2)
                C[i, j] += alpha * A[i, k] * B[k, j]
            end
        end
    end
    return C
end

function kernel_gemmver(alpha, beta, u1, u2, v1, v2, A, x, y, z)
    w = zeros(eltype(A), axes(A, 1))
    for i in axes(A, 1)
        for j in axes(A, 2)
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
        end
    end

    for i in axes(A, 1)
        for j in axes(A, 2)
            x[i] = x[i] + beta * A[j, i] * y[j]
        end
    end

    for i in axes(A, 1)
        x[i] = x[i] + z[i]
    end

    for i in axes(A, 1)
        for j in axes(A, 2)
            w[i] = w[i] + alpha * A[i, j] * x[j]
        end
    end
    return w
end

function kernel_gesummv(alpha, beta, A, B, x)
    tmp = zeros(eltype(A), axes(A, 1))
    y = zeros(eltype(A), axes(A, 1))
    for i in axes(A, 1)
        tmp[i] = 0.0
        y[i] = 0.0
        for j in axes(A, 2)
            tmp[i] = A[i, j] * x[j] + tmp[i]
            y[i] = B[i, j] * x[j] + y[i]
        end
        y[i] = alpha * tmp[i] + beta * y[i]
    end
    return y
end

function kernel_symm(alpha, beta, A, B)
    C = zeros(eltype(A), axes(A, 1), axes(B, 2))
    for i in axes(A, 1)
        for j in axes(B, 2)
            temp2 = 0.0
            for k in axes(A, 2)
                C[k, j] += alpha * B[i, j] * A[i, k]
                temp2 += B[k, j] * A[i, k]
            end
            C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2
        end
    end
    return C
end

function kernel_syr2k(alpha, beta, A, B)
    C = zeros(eltype(A), axes(A, 1), axes(A, 1))
    for i in axes(A, 1)
        for j in 1:i
            C[i, j] *= beta
        end
        for k in axes(A, 2)
            for j in 1:i
                C[i, j] += A[j, k] * alpha * B[i, k] + B[j, k] * alpha * A[i, k]
            end
        end
    end
    return C
end

function kernel_syrk(alpha, beta, A)
    C = zeros(eltype(A), axes(A, 1), axes(A, 1))
    for i in axes(A, 1)
        for j in 1:i
            C[i, j] *= beta
        end
        for k in axes(A, 2)
            for j in 1:i
                C[i, j] += alpha * A[i, k] * A[j, k]
            end
        end
    end
    return C
end

function kernel_trmm(alpha, A, B)
    for i in axes(A, 1)
        for j in axes(B, 2)
            for k in axes(A, 2)
                B[i, j] += A[k, i] * B[k, j]
            end
            B[i, j] = alpha * B[i, j]
        end
    end
    return B
end

function kernel_2mm(alpha, beta, A, B, C, D)
    tmp = zeros(eltype(A), axes(A, 1), axes(B, 2))
    for i in axes(A, 1)
        for j in axes(B, 2)
            tmp[i, j] = 0
            for k in axes(A, 2)
                tmp[i, j] += alpha * A[i, k] * B[k, j]
            end
        end
    end
    for i in axes(A, 1)
        for j in axes(C, 2)
            D[i, j] *= beta
            for k in axes(B, 1)
                D[i, j] += tmp[i, k] * C[k, j]
            end
        end
    end
    return D
end

function kernel_3mm(A, B, C, D)
    E = zeros(eltype(A), axes(A, 1), axes(A, 2))
    F = zeros(eltype(A), axes(A, 1), axes(A, 2))
    G = zeros(eltype(A), axes(A, 1), axes(A, 2))
    for i in axes(A, 1)
        for j in axes(B, 2)
            E[i, j] = 0.0
            for k in axes(A, 2)
                E[i, j] += A[i, k] * B[k, j]
            end
        end
    end

    for i in axes(C, 1)
        for j in axes(D, 2)
            F[i, j] = 0.0
            for k in axes(C, 2)
                F[i, j] += C[i, k] * D[k, j]
            end
        end
    end

    for i in axes(E, 1)
        for j in axes(F, 2)
            G[i, j] = 0.0
            for k in axes(E, 2)
                G[i, j] += E[i, k] * F[k, j]
            end
        end
    end
    return G
end

function kernel_atax(A, x)
    tmp = zeros(eltype(A), axes(A, 1))
    y = zeros(eltype(A), axes(A, 2))

    for i in axes(A, 2)
        y[i] = 0
    end
    for i in axes(A, 1)
        tmp[i] = 0
        for j in axes(A, 2)
            tmp[i] = tmp[i] + A[i, j] * x[j]
        end
        for j in axes(A, 2)
            y[j] = y[j] + A[i, j] * tmp[i]
        end
    end
    return y
end

function kernel_bicg(A, p, r)
    s = zeros(eltype(A), axes(A, 2))
    q = zeros(eltype(A), axes(A, 1))

    for i in axes(A, 2)
        s[i] = 0
    end
    for i in axes(A, 1)
        q[i] = 0
        for j in axes(A, 2)
            s[j] = s[j] + r[i] * A[i, j]
            q[i] = q[i] + A[i, j] * p[j]
        end
    end
    return s, q
end

function kernel_doitgen(A, C4)
    sum = similar(A, size(A, 2))
    for r in axes(A, 1)
        for q in axes(A, 2)
            for p in axes(A, 2)
                sum[p] = 0.0
                for s in axes(A, 2)
                    sum[p] += A[r, q, s] * C4[s, p]
                end
            end
            for p in axes(A, 2)
                A[r, q, p] = sum[p]
            end
        end
    end
    return A
end

function kernel_mvt(x1, x2, y1, y2, A)
    for i in axes(x1, 1)
        for j in axes(y1, 1)
            x1[i] += A[i, j] * y1[j]
        end
    end

    for i in axes(x2, 1)
        for j in axes(y2, 1)
            x2[i] += A[i, j] * y2[j]
        end
    end

    return x1, x2
end

function kernel_cholesky(A)
    for i in axes(A, 1)
        for j in 1:(i - 1)
            for k in 1:(j - 1)
                A[i, j] -= A[i, k] * A[j, k]
            end
            A[i, j] /= A[j, j]
        end

        for k in 1:(i - 1)
            A[i, i] -= A[i, k] * A[i, k]
        end
        A[i, i] = sqrt(A[i, i])
    end
    return A
end

function kernel_durbin(r, y)
    y[1] = -r[1]
    beta = 1.0
    alpha = -r[1]
    z = zero(y)
    for k in 2:size(y, 1)
        beta = (1 - alpha * alpha) * beta
        sum = 0.0
        for i in 1:k
            sum += r[k - i] * y[i]
        end
        alpha = -(r[k] + sum) / beta

        for i in 1:k
            z[i] = y[i] + alpha * y[k - i]
        end

        for i in 1:k
            y[i] = z[i]
        end
        y[k] = alpha
    end
    return y
end

function kernel_gramschidt(A, R, Q)
    for k in axes(A, 2)
        nrm = 0.0
        for i in axes(A, 1)
            nrm += A[i, k]^2
        end
        R[k, k] = sqrt(nrm)
        for i in axes(Q, 1)
            Q[i, k] = A[i, k] / R[k, k]
        end

        for j in (k + 1):size(R, 1)
            R[k, j] = 0.0
            for i in axes(A, 1)
                R[k, j] += Q[i, k] * A[i, j]
            end
            for i in axes(A, 1)
                A[i, j] -= Q[i, k] * R[k, j]
            end
        end
    end
    return A
end

function kernel_lu(A)
    for i in axes(A, 1)
        for j in 1:(i - 1)
            for k in 1:(j - 1)
                A[i, j] -= A[i, k] * A[k, j]
            end
            A[i, j] /= A[j, j]
        end
        for j in axes(A, 1)
            for k in 1:(i - 1)
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    return A
end

function kernel_ludcmp(A, b, x, y)
    for i in axes(A, 1)
        for j in 1:(i - 1)
            w = A[i, j]
            for k in 1:(j - 1)
                w -= A[i, k] * A[k, j]
            end
            A[i, j] = w / A[j, j]
        end
        for j in 1:(i - 1)
            w = A[i, j]
            for k in 1:(j - 1)
                w -= A[i, k] * A[k, j]
            end
            A[i, j] = w
        end
    end

    for i in axes(b, 1)
        w = b[i]
        for j in 1:(i - 1)
            w -= A[i, j] * y[j]
        end
        y[i] = w
    end

    for i in size(y, 1):-1:1
        w = y[i]
        for j in (i + 1):size(A, 2)
            w -= A[i, j] * x[j]
        end
        x[i] = w / A[i, i]
    end
    return A
end

function kernel_trisolv(L, x, b)
    for i in axes(x, 1)
        x[i] = b[i]
        for j in 1:(i - 1)
            x[i] -= L[i, j] * x[j]
        end
        x[i] = x[i] / L[i, i]
    end
    return x
end

function kernel_adi(tsteps::Int, u, v, p, q)
    N = size(u, 1)
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / tsteps
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in 1:tsteps
        #//Column Sweep
        for i in 2:(N - 1)
            v[1, i] = 1.0
            p[i, 1] = 0.0
            q[i, 1] = v[1, i]
            for j in 2:(N - 1)
                p[i, j] = -c / (a * p[i, j - 1] + b)
                q[i, j] =
                    (
                        -d * u[j, i - 1] + (1.0 + 2.0 * d) * u[j, i] - f * u[j, i + 1] -
                        a * q[i, j - 1]
                    ) / (a * p[i, j - 1] + b)
            end

            v[N, i] = 1.0
            for j in (N - 1):-1:2
                v[j, i] = p[i, j] * v[j + 1, i] + q[i, j]
            end
        end
        #//Row Sweep
        for i in 2:(N - 1)
            u[i, 1] = 1.0
            p[i, 1] = 0.0
            q[i, 1] = u[i, 1]
            for j in 2:(N - 1)
                p[i, j] = -f / (d * p[i, j - 1] + e)
                q[i, j] =
                    (
                        -a * v[i - 1, j] + (1.0 + 2.0 * a) * v[i, j] - c * v[i + 1, j] -
                        d * q[i, j - 1]
                    ) / (d * p[i, j - 1] + e)
            end
            u[i, N] = 1.0
            for j in (N - 1):-1:2
                u[i, j] = p[i, j] * u[i, j + 1] + q[i, j]
            end
        end
    end
    return u, v
end

function kernel_fdtd_2d(EX, EY, HZ, fict)
    for t in axes(fict, 1)
        for j in axes(EY, 2)
            EY[1, j] = fict[t]
        end
        for i in 2:size(EY, 1)
            for j in axes(EY, 2)
                EY[i, j] = EY[i, j] - 0.5 * (HZ[i, j] - HZ[i - 1, j])
            end
        end
        for i in axes(EX, 1)
            for j in 2:size(EX, 2)
                EX[i, j] = EX[i, j] - 0.5 * (HZ[i, j] - HZ[i, j - 1])
            end
        end
        for i in 1:(size(HZ, 1) - 1)
            for j in 1:(size(HZ, 2) - 1)
                HZ[i, j] =
                    HZ[i, j] - 0.7 * (EX[i, j + 1] - EX[i, j] + EY[i + 1, j] - EY[i, j])
            end
        end
    end

    return HZ
end

function kernel_heat_3d(tsteps::Int, A, B)
    N = size(A, 1)

    for t in 1:tsteps
        for i in 2:(N - 1)
            for j in 2:(N - 1)
                for k in 2:(N - 1)
                    B[i, j, k] =
                        0.125 * (A[i + 1, j, k] - 2.0 * A[i, j, k] + A[i - 1, j, k])
                    +0.125 * (A[i, j + 1, k] - 2.0 * A[i, j, k] + A[i, j - 1, k])
                    +0.125 * (A[i, j, k + 1] - 2.0 * A[i, j, k] + A[i, j, k - 1])
                    +A[i, j, k]
                end
            end
        end
        for i in 2:(N - 1)
            for j in 2:(N - 1)
                for k in 2:(N - 1)
                    A[i, j, k] =
                        0.125 * (B[i + 1, j, k] - 2.0 * B[i, j, k] + B[i - 1, j, k])
                    +0.125 * (B[i, j + 1, k] - 2.0 * B[i, j, k] + B[i, j - 1, k])
                    +0.125 * (B[i, j, k + 1] - 2.0 * B[i, j, k] + B[i, j, k - 1])
                    +B[i, j, k]
                end
            end
        end
    end
    return A, B
end

function kernel_jacobi_1d(tsteps::Int, A, B)
    N = size(A, 1)
    for t in 1:tsteps
        for i in 2:(N - 1)
            B[i] = 1 / 3 * (A[i - 1] + A[i] + A[i + 1])
        end
        for i in 2:(N - 1)
            A[i] = 1 / 3 * (B[i - 1] + B[i] + B[i + 1])
        end
    end
    return A, B
end

function kernel_jacobi_2d(tsteps::Int, A, B)
    N = size(A, 1)

    for t in 1:tsteps
        for i in 2:(N - 1)
            for j in 2:(N - 1)
                B[i, j] =
                    0.2 * (A[i, j] + A[i, j - 1] + A[i, 1 + j] + A[1 + i, j] + A[i - 1, j])
            end
        end
        for i in 2:(N - 1)
            for j in 2:(N - 1)
                A[i, j] =
                    0.2 * (B[i, j] + B[i, j - 1] + B[i, 1 + j] + B[1 + i, j] + B[i - 1, j])
            end
        end
    end
    return A, B
end

function kernel_seidel_2d(tsteps::Int, A)
    N = size(A, 1)
    for _ in 1:tsteps
        for i in 2:(N - 1)
            for j in 2:(N - 1)
                A[i, j] =
                    (
                        A[i - 1, j - 1] +
                        A[i - 1, j] +
                        A[i - 1, j + 1] +
                        A[i, j - 1] +
                        A[i, j] +
                        A[i, j + 1] +
                        A[i + 1, j - 1] +
                        A[i + 1, j] +
                        A[i + 1, j + 1]
                    ) / 9.0
            end
        end
    end
    return A
end

function kernel_deriche(alpha, I)
    Y1 = zero(I)
    Y2 = zero(I)
    O = zero(I)
    k = (1.0 - exp(-alpha))^2 / (1.0 + 2.0 * exp(-alpha) - exp(2.0 * alpha))
    a1 = a5 = k
    a2 = a6 = k * exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * exp(-2.0 * alpha)
    b1 = 2.0^(-alpha)
    b2 = -exp(-2.0 * alpha)
    c1 = c2 = 1

    for i in axes(I, 1)
        ym1 = ym2 = xm1 = 0.0
        for j in axes(I, 2)
            Y1[i, j] = a1 * I[i, j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = I[i, j]
            ym2 = ym1
            ym1 = Y1[i, j]
        end
    end

    for i in axes(I, 1)
        yp1 = yp2 = xp1 = xp2 = 0.0
        for j in size(I, 2):-1:1
            Y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = I[i, j]
            yp1 = Y2[i, j]
        end
    end

    for i in axes(I, 1)
        for j in axes(I, 2)
            O[i, j] = c1 * (Y1[i, j] + Y2[i, j])
        end
    end

    for j in axes(I, 2)
        tm1 = ym1 = ym2 = 0.0
        for i in axes(I, 1)
            Y1[i, j] = a5 * O[i, j] + a6 * tm1 + b1 * ym1 + b2 * ym2
            tm1 = O[i, j]
            ym2 = ym1
            ym1 = Y1[i, j]
        end
    end

    for j in axes(I, 2)
        tp1 = tp2 = yp1 = yp2 = 0.0
        for i in size(I, 1):-1:1
            Y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tp2 = tp1
            tp1 = O[i, j]
            yp2 = yp1
            yp1 = Y2[i, j]
        end
    end

    for i in axes(I, 1)
        for j in axes(I, 2)
            O[i, j] = c2 * (Y1[i, j] + Y2[i, j])
        end
    end
    return O
end

function kernel_floyd_warshall(path)
    for k in axes(path, 1)
        for i in axes(path, 1)
            for j in axes(path, 1)
                path[i, j] = if path[i, j] < path[i, k] + path[k, j]
                    path[i, j]
                else
                    path[i, k] + path[k, j]
                end
            end
        end
    end
    return path
end

function kernel_nussinov(seq, T)
    for i in size(seq, 1):-1:1
        for j in (i + 1):size(seq, 1)
            if j >= 1
                T[i, j] = max(T[i, j], T[i, j - 1])
            end

            if i + 1 <= size(seq, 1)
                T[i, j] = max(T[i, j], T[i + 1, j])
            end

            if j >= 1 && (i + 1 <= size(seq, 1))
                if i < j - 1
                    tmp = (((seq[i] + seq[j]) - 3.0) < 10e-5) ? 1.0 : 0.0
                    T[i, j] = max(T[i, j], T[i + 1, j - 1] + tmp)
                else
                    T[i, j] = max(T[i, j], T[i + 1, j - 1])
                end
            end

            for k in (i + 1):(j - 1)
                T[i, j] = max(T[i, j], T[i, j] + T[k + 1, j])
            end
        end
    end
    return T
end

end

function kernel_deriche2(I)
    Y1 = zero(I)
        ym1  = 0.0
        ym2 = 0.0
        i = 1
        for j in axes(I, 2)
            Y1[i, j] = I[i, j] +  ym1 + ym2
            ym2 = ym1 + 1.0
            ym1 = Y1[i, j]
        end
    return Y1
end

@testset "Polybench" begin
    A = collect(reshape(1:1.0:256, 16, 16))
    c = collect(reshape(1:16, 1, 16))
    tA = Reactant.to_rarray(A)
    f = collect(reshape(0:0.065:1, 1, 16))
    tf = Reactant.to_rarray(f)

    @jit(kernel_correlation(tA))
    @jit(kernel_correlation(tA))

    @test @jit(kernel_gemm(5, 2, tA, tA)) == kernel_gemm(5, 2, A, A)
    @test @jit(kernel_gemmver(2.0, 0.01, tf, tf, tf, tf, tA, tf, tf, tf)) ≈
        kernel_gemmver(2.0, 0.01, f, f, f, f, A, f, f, f)
    @test @jit(kernel_gesummv(2.0, 0.01, tA, tA, tf)) ≈ kernel_gesummv(2.0, 0.01, A, A, f)
    @test @jit(kernel_symm(2.0, 0.01, tA, tA)) ≈ kernel_symm(2.0, 0.01, A, A)
    @test @jit(kernel_syr2k(2.0, 0.01, tA, tA)) ≈ kernel_syr2k(2.0, 0.01, A, A)
    @test @jit(kernel_syrk(2.0, 0.01, tA)) ≈ kernel_syrk(2.0, 0.01, A)
    @test @jit(kernel_trmm(2.0, tA, tA)) ≈ kernel_trmm(2.0, A, A)
    @test @jit(kernel_2mm(2.0, 0.01, tA, tA, tA, tA)) ≈ kernel_2mm(2.0, 0.01, A, A, A, A)
    @test @jit(kernel_3mm(tA, tA, tA, tA)) ≈ kernel_3mm(A, A, A, A)
    @test @jit(kernel_atax(tA, tf)) ≈ kernel_atax(A, f)
    @test all((@jit(kernel_bicg(tA, tf, tf)) .≈ kernel_bicg(A, f, f)))
    AA = ones(16, 16, 16)
    tAA = Reactant.to_rarray(AA)
    @test @jit((kernel_doitgen(tAA, tA))) ≈ kernel_doitgen(AA, A)
    @test all((@jit(kernel_mvt(tf, tf, tf, tf, tA)) .≈ kernel_mvt(f, f, f, f, A)))
    @test @jit(kernel_cholesky(tA)) ≈ kernel_cholesky(A)
    @test @jit(kernel_durbin(tf, tf)) ≈ kernel_durbin(f, f)
    @test @jit(kernel_gramschidt(tA, tA, tA)) ≈ kernel_gramschidt(A, A, A)
    @test @jit(kernel_lu(tA)) ≈ kernel_lu(A)
    @test @jit(kernel_ludcmp(tA, tA, tf, tf)) ≈ kernel_ludcmp(A, A, f, f)
    @test @jit(kernel_trisolv(tA, tf, tf)) ≈ kernel_trisolv(A, f, f)
    tA1 = Reactant.to_rarray(A)
    tA2 = Reactant.to_rarray(A)
    tA3 = Reactant.to_rarray(A)
    tA4 = Reactant.to_rarray(A)
    @test all(
        @jit(kernel_adi(5, tA1, tA2, tA3, tA4)) .≈
        kernel_adi(5, copy(A), copy(A), copy(A), copy(A)),
    )
    @test @jit(kernel_fdtd_2d(tA, tA, tA, tf)) ≈ kernel_fdtd_2d(A, A, A, f)
    @test all(@jit(kernel_heat_3d(5, tAA, tAA)) .≈ kernel_heat_3d(5, AA, AA))
    @test all(@jit(kernel_jacobi_1d(5, tf, tf)) .≈ kernel_jacobi_1d(5, f, f))
    @test all(@jit(kernel_jacobi_2d(5, tA, tA)) .≈ kernel_jacobi_2d(5, A, A))
    @test all(@jit(kernel_seidel_2d(5, tA)) .≈ kernel_seidel_2d(5, A))
    @test all(@jit(kernel_deriche(0.1, tA)) .≈ kernel_deriche(0.1, A)) #TODO: FAIL
    @test @jit(kernel_floyd_warshall(tA)) == kernel_floyd_warshall(A)
    @test @jit(kernel_nussinov(tf, tA)) ≈ kernel_nussinov(f, A)
end