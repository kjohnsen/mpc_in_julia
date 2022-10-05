
using LinearAlgebra
using Optimization, OptimizationOptimJL, ForwardDiff

#define system
A = [1 -6.66e-13 -2.03e-9 -4.14e-6;
    9.83e-4 1 -4.09e-8 -8.32e-5;
    4.83e-7 9.83e-4 1 -5.34e-4;
    1.58e-10 4.83e-7 9.83e-4 .9994;
]
B = [9.83e-4 4.83e-7 1.58e-10 3.89e-14]'
C = [-.0096 .0135 .005 -.0095]
D = [0.0]
T = 3
Q = C'*C
R = [1e-4]

#for getting firing rate Z - y[1] used as julia gives a vector for y during following processes by default
function y_to_z(y)
    return exp(61.4*y[1] - 5.468);
end


unpack_u(u) = [reshape(u[1:4*T], (4, T)), reshape(u[1+end-T:end], (1, T))]
XU_to_u(X, U) = reduce(vcat, [X[:], U[:]])

u0 = XU_to_u(zeros(4, T), zeros(1, T))

function J(u, p)
    X, U = unpack_u(u)
    xref = p[1]
    X̃ = X .- xref
    return sum((Q*X̃).*X̃) + sum((R*U).*U)
end
function cons(u, p)
    X, U = unpack_u(u)
    x0 = p[2]
    return reduce(vcat, [
        (X[:, 1] - x0)[:],
        (X[:, 2:end] - (A*X[:, 1:end-1] + B*U[:, 1:end-1]))[:]  # to vector
    ])
end 
optf = OptimizationFunction(J, Optimization.AutoForwardDiff(), cons=cons)

# static reference 0.2 for now
zref(t) = .2

function u0p_from_prev(u, xref, uref)
    X, U = unpack_u(u)
    newX = A*X + B*U
    newU = zero(U)
    newU[:, 1:end-1] = U[:, 2:end]
    newU[:, end] = uref
    # p is [xref, x0]
    return XU_to_u(newX, newU), [xref, newX[:, 1]]
end

solu = u0
N = 3
Z = zeros(1, N)
U = zeros(1, N)
for t in 1:N
    yref = (log(zref(t)) + 5.468)/61.4
    uref = inv(C*inv(I-A)*B) * yd
    xref = inv(I-A) * B * ud
    u0, p = u0p_from_prev(solu, xref, uref)
    # println(u0[end-T+3])

    prob = OptimizationProblem(optf, u0, p, lcons=zeros(5*T), ucons=zeros(5*T))
    sol = solve(prob, BFGS())
    println(optf.cons(sol.u, p))

    Z[1, t] = y_to_z(C*p[2])  # p[2] is x0
    U[1, t] = sol.u[end-T+1]
end


using Plots
begin
    plot(Z', label="z")
    plot!(U', label="u")
end