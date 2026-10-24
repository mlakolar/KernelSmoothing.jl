## Cluster-robust variances

# The estimates and cluster-robust standard errors at x0, as for the other variance
# estimators; with `vectors = true` also the sums scale √c ∑_{i ∈ g} ℓᵢ ε̃ᵢ and their
# bias-corrected counterparts over the clusters g in the window, whose numbers are
# `keys`. The adjustments of types 2 and 3 use yᵢ = L⁻¹ r_p(uᵢ) with Γ_p = L L', so
# that the hat matrix of the fit of order p, in terms of the residuals, is Y Y' Kh.
function _localfit(d::SmoothingData, k::AbstractKernel, x0, h, b, p, ν, vce::ClusterVariance;
                   vectors::Bool = false, setup = nothing, compat::Bool = false)
  f = setup === nothing ? _localsetup(d, k, x0, h, b, p, ν) : setup
  a, e = f.a, f.e
  m = e - a + 1   # the observations with positive weight for the larger bandwidth
  if !f.ok
    fit = _LocalFit(NaN, NaN, NaN, NaN, m, f.nuniq)
    vectors || return fit
    keys = m > 0 ? sort!(unique(view(d.cluster, a:e))) : Int[]
    return (fit = fit, keys = keys, aus = fill(NaN, length(keys)), arb = fill(NaN, length(keys)))
  end
  xs, ys = d.x, d.y
  βp, γ, g, c, ρ, s, scale = f.βp, f.γ, f.g, f.c, f.ρ, f.s, f.scale
  P = p + 1
  adjust = vce.type >= 2
  # nprobust's adjustment of the bias-corrected residuals needs the bias weights and Λ
  nb = adjust && compat
  ℓ, ℓbc, εh, εb = (Vector{Float64}(undef, m) for _ in 1:4)
  Y = adjust ? Matrix{Float64}(undef, P, m) : nothing
  KY = adjust ? Matrix{Float64}(undef, P, m) : nothing
  kc = nb ? Vector{Float64}(undef, m) : nothing
  Λ = nb ? zeros(P) : nothing
  U = adjust ? f.Cp.U : nothing
  for i in a:e
    j = i - a + 1
    t = xs[i] - x0
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    ℓ[j] = kh * evalpoly(u, g)
    ℓbc[j] = ℓ[j] - ρ * s * kb * evalpoly(v, c)
    εh[j] = ys[i] - evalpoly(u, βp)
    εb[j] = ys[i] - evalpoly(v, γ)
    if adjust
      y = _powers!(view(Y, :, j), u)
      if nb
        _addpowers!(Λ, kh * u^P, u)   # Λ = ∑ khᵢ uᵢ^(p+1) r_p(uᵢ)
        kc[j] = kb * evalpoly(v, c)
      end
      _lsolve!(y, U)
      KY[:, j] .= kh .* y
    end
  end
  cl = view(d.cluster, a:e)
  keys, sh = _clustersums(cl, ℓ, εh, Y, KY, vce.type, compat)
  sb = if nb
    # nprobust's matrix for the bias-corrected residuals, Q Γ⁻¹ Q' in its variables, is
    # q̃ᵢ'Γ_p⁻¹q̃ⱼ / h with q̃ᵢ = khᵢ r_p(uᵢ) - ρ kbᵢ c'r_q(vᵢ) Λ: its 1/h, from the
    # kernel weights k(u)/h, makes it change with the units of x
    z = _lsolve!(copy(Λ), U)
    A = (KY .- ρ .* z .* kc') ./ sqrt(h)
    last(_clustersums(cl, ℓbc, εb, A, A, vce.type, compat))
  else
    last(_clustersums(cl, ℓbc, εb, Y, KY, vce.type, compat))
  end
  G = length(keys)
  ch, cb = _crscale(vce.type, m, G, p + 1, compat), _crscale(vce.type, m, G, p + 2, compat)
  vh, vb = ch * sum(abs2, sh), cb * sum(abs2, sb)
  fit = _LocalFit(scale * βp[ν + 1], scale * sqrt(vh), scale * (βp[ν + 1] - ρ * s * γ[p + 2]),
                  scale * sqrt(vb), m, f.nuniq)
  vectors || return fit
  return (fit = fit, keys = keys, aus = (scale * sqrt(ch)) .* sh, arb = (scale * sqrt(cb)) .* sb)
end

# The sums S_g = ∑_{i ∈ g} ℓᵢ ε̃ᵢ over the clusters g of the observations, numbered by
# cl, in increasing order of their numbers, which are returned with them. For types
# 2 and 3, ε̃_g = f(I - A_g' B_g) ε_g with f(s) = s^(-1/2) or 1/s, where column i of A
# and B belongs to observation i. Since f(I - A'B) = I + A'φ(BA')B with
# φ(λ) = (f(1 - λ) - 1) / λ, S_g = ∑ ℓᵢ εᵢ + (∑ ℓᵢ aᵢ)'φ(K)(∑ bᵢ εᵢ) with K = ∑ bᵢ aᵢ',
# so that each cluster needs the eigendecomposition of a small matrix only, whatever
# its size; K is symmetric for the matrices used here.
function _clustersums(cl, ℓ, ε, A, B, type, compat)
  ord = sortperm(cl)
  keys, sums = Int[], Float64[]
  adjust = type >= 2
  P = adjust ? size(A, 1) : 0
  K, la, w = zeros(P, P), zeros(P), zeros(P)
  i, n = 1, length(ord)
  while i <= n
    key = cl[ord[i]]
    S = 0.0
    if adjust
      fill!(K, 0.0); fill!(la, 0.0); fill!(w, 0.0)
    end
    while i <= n && cl[ord[i]] == key
      j = ord[i]
      S += ℓ[j] * ε[j]
      if adjust
        for r in 1:P
          la[r] += ℓ[j] * A[r, j]
          w[r] += B[r, j] * ε[j]
          for q in r:P
            K[r, q] += B[r, j] * A[q, j]
          end
        end
      end
      i += 1
    end
    adjust && (S += _cradjustment(K, la, w, type, compat, n > P))
    push!(keys, key)
    push!(sums, S)
  end
  return keys, sums
end

# la'φ(K)w for the symmetric K, from its eigenvalues λ; the eigenvalues of I - H_gg
# are s = 1 - λ and, where the cluster has more observations than K has rows, one
function _cradjustment(K, la, w, type, compat, ones)
  E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(K))
  λ, V = E.values, E.vectors
  s = 1 .- λ
  if compat
    # nprobust raises the eigenvalues of I - H_gg below 1e-12 to 1e-12 for CR2, and
    # for CR3 where solve finds I - H_gg singular, with a reciprocal condition number
    # below the machine epsilon
    lo, hi = minimum(abs, s), maximum(abs, s)
    raise = type == 2 || (ones ? min(lo, 1.0) / max(hi, 1.0) : lo / hi) < eps()
  else
    # a cluster that alone determines part of the fit leaves no residual to estimate
    # its variance from, as a leverage of one does for HCVariance
    minimum(s) > sqrt(eps()) || return NaN
    raise = false
  end
  t = 0.0
  for q in eachindex(λ)
    sq = raise ? max(s[q], 1e-12) : s[q]
    φ = sq == s[q] ? (type == 2 ? 1 / (sqrt(sq) * (1 + sqrt(sq))) : 1 / sq) :
                     ((type == 2 ? 1 / sqrt(sq) : 1 / sq) - 1) / λ[q]
    vq = view(V, :, q)
    t += φ * LinearAlgebra.dot(vq, la) * LinearAlgebra.dot(vq, w)
  end
  return t
end

# The factor c of the cluster-robust variance with n observations and G clusters in
# the window and k coefficients: NaN without two clusters, or for CR1 without more
# observations than coefficients, except under compat, where nprobust's arithmetic
# gives Inf or NaN
function _crscale(type, n, G, k, compat)
  compat || (G >= 2 && (type != 1 || n > k)) || return NaN
  type == 1 && return (n - 1) / (n - k) * (G / (G - 1))
  type == 3 && return (G - 1) / G
  return 1.0
end

# y = (1, u, …, u^(length(y) - 1))
function _powers!(y, u)
  v = 1.0
  for j in eachindex(y)
    y[j] = v
    v *= u
  end
  return y
end

# adds t u^(j-1) to y[j]
function _addpowers!(y, t, u)
  for j in eachindex(y)
    y[j] += t
    t *= u
  end
  return y
end

# y = L⁻¹ y in place for the Cholesky factor U = L' of Γ = U'U
function _lsolve!(y, U)
  for i in eachindex(y)
    s = y[i]
    for j in 1:i-1
      s -= U[j, i] * y[j]
    end
    y[i] = s / U[i, i]
  end
  return y
end

# C[j, l] = ∑_g A[j][g] A[l][g] over the clusters g in both keys[j] and keys[l], which
# are sorted
function _clustergram(keys, vals)
  G = length(keys)
  C = zeros(G, G)
  for j in 1:G, l in j:G
    kj, kl = keys[j], keys[l]
    a, b, s = 1, 1, 0.0
    while a <= length(kj) && b <= length(kl)
      if kj[a] == kl[b]
        s += vals[j][a] * vals[l][b]
        a += 1
        b += 1
      elseif kj[a] < kl[b]
        a += 1
      else
        b += 1
      end
    end
    C[j, l] = C[l, j] = s
  end
  return C
end

# The cluster-robust counterpart of _coefvar, c ∑_g (∑_{i ∈ g} wᵢ k(uᵢ) g'r(uᵢ) ε̃ᵢ)²
# over the window of a fit, from its residuals; under compat, as nprobust's
# lprobust.bw, the residuals of HCVariance(type) with the factor c of CR1
function _coefvar(d, k, f, g, vce::ClusterVariance, compat::Bool)
  a, e, h, o, c = f.a, f.e, f.h, f.o, f.c
  m, P = e - a + 1, o + 1
  adjust = !compat && vce.type >= 2
  lev = compat && vce.type >= 2 ? _leveragepoly(f.C, o) : nothing
  ℓ, ε = Vector{Float64}(undef, m), Vector{Float64}(undef, m)
  Y = adjust ? Matrix{Float64}(undef, P, m) : nothing
  KY = adjust ? Matrix{Float64}(undef, P, m) : nothing
  U = adjust ? f.C.U : nothing
  for i in a:e
    j = i - a + 1
    u = (d.x[i] - c) / h
    kw = _weight(d.w, i) * k(u)
    ℓ[j] = kw * evalpoly(u, g)
    εi = d.y[i] - evalpoly(u, f.β)
    if compat && vce.type == 1
      εi *= _dofscale(m, P)
    elseif compat && vce.type >= 2
      hii = kw * evalpoly(u, lev)
      εi = 1 - hii > sqrt(eps()) ? (vce.type == 2 ? εi / sqrt(1 - hii) : εi / (1 - hii)) : NaN
    end
    ε[j] = εi
    if adjust
      y = _lsolve!(_powers!(view(Y, :, j), u), U)
      KY[:, j] .= kw .* y
    end
  end
  type = compat ? 1 : vce.type
  keys, sums = _clustersums(view(d.cluster, a:e), ℓ, ε, Y, KY, type, compat)
  return _crscale(type, m, length(keys), P, compat) * sum(abs2, sums)
end
