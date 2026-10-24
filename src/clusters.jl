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
        t = kh * u^P   # Λ = ∑ khᵢ uᵢ^(p+1) r_p(uᵢ)
        for r in 1:P
          Λ[r] += t
          t *= u
        end
        kc[j] = kb * evalpoly(v, c)
      end
      _lsolve!(y, U)
      KY[:, j] .= kh .* y
    end
  end
  cl = view(d.cluster, a:e)
  ord = _clusterorder(cl)
  keys, sh = _clustersums(cl, ord, ℓ, εh, Y, KY, vce.type, compat)
  sb = if nb
    # nprobust's matrix for the bias-corrected residuals, Q Γ⁻¹ Q' in its variables, is
    # q̃ᵢ'Γ_p⁻¹q̃ⱼ / h with q̃ᵢ = khᵢ r_p(uᵢ) - ρ kbᵢ c'r_q(vᵢ) Λ: its 1/h, from the
    # kernel weights k(u)/h, makes it change with the units of x
    z = _lsolve!(copy(Λ), U)
    A = (KY .- ρ .* z .* kc') ./ sqrt(h)
    last(_clustersums(cl, ord, ℓbc, εb, A, A, vce.type, compat))
  else
    last(_clustersums(cl, ord, ℓbc, εb, Y, KY, vce.type, compat))
  end
  G = length(keys)
  ch, cb = _crscale(vce.type, m, G, p + 1, compat), _crscale(vce.type, m, G, p + 2, compat)
  vh, vb = ch * sum(abs2, sh), cb * sum(abs2, sb)
  fit = _LocalFit(scale * βp[ν + 1], scale * sqrt(vh), scale * (βp[ν + 1] - ρ * s * γ[p + 2]),
                  scale * sqrt(vb), m, f.nuniq)
  vectors || return fit
  return (fit = fit, keys = keys, aus = (scale * sqrt(ch)) .* sh, arb = (scale * sqrt(cb)) .* sb)
end

# The order of the observations by their clusters cl: a counting sort over the
# numbers of the clusters, unless they range far beyond the observations
function _clusterorder(cl)
  m = length(cl)
  G = m == 0 ? 0 : maximum(cl)
  G > 4m && return sortperm(cl)
  next = zeros(Int, G + 1)
  for c in cl
    next[c + 1] += 1
  end
  next[1] = 1
  for c in 2:G+1
    next[c] += next[c - 1]   # next[c] is where cluster c starts in the order
  end
  ord = Vector{Int}(undef, m)
  for (j, c) in enumerate(cl)
    ord[next[c]] = j
    next[c] += 1
  end
  return ord
end

# The sums S_g = ∑_{i ∈ g} ℓᵢ ε̃ᵢ over the clusters g of the observations, numbered by
# cl and taken in the order ord, which sorts them by cluster, in increasing order of
# their numbers, which are returned with them. For types
# 2 and 3, ε̃_g = f(I - A_g' B_g) ε_g with f(s) = s^(-1/2) or 1/s, where column i of A
# and B belongs to observation i. Since f(I - A'B) = I + A'φ(BA')B with
# φ(λ) = (f(1 - λ) - 1) / λ, S_g = ∑ ℓᵢ εᵢ + (∑ ℓᵢ aᵢ)'φ(K)(∑ bᵢ εᵢ) with K = ∑ bᵢ aᵢ',
# so that each cluster needs the eigendecomposition of a small matrix only, whatever
# its size; K is symmetric for the matrices used here.
function _clustersums(cl, ord, ℓ, ε, A, B, type, compat)
  keys, sums = Int[], Float64[]
  adjust = type >= 2
  P = adjust ? size(A, 1) : 0
  K, la, w = zeros(P, P), zeros(P), zeros(P)
  λ, V = zeros(P), zeros(P, P)
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
    adjust && (S += _cradjustment!(K, la, w, type, compat, n > P, λ, V))
    push!(keys, key)
    push!(sums, S)
  end
  return keys, sums
end

# la'φ(K)w for the symmetric K, whose upper triangle is used and which is
# overwritten, from its eigenvalues λ and eigenvectors V, which are overwritten too.
# The eigenvalues of I - H_gg are s = 1 - λ and, where the cluster has more
# observations than K has rows, one.
function _cradjustment!(K, la, w, type, compat, ones, λ, V)
  _jacobi!(λ, V, K)
  smin, lo, hi = Inf, Inf, 0.0
  for q in eachindex(λ)
    sq = 1 - λ[q]
    smin, lo, hi = min(smin, sq), min(lo, abs(sq)), max(hi, abs(sq))
  end
  if compat
    # nprobust raises the eigenvalues of I - H_gg below 1e-12 to 1e-12 for CR2, and
    # for CR3 where solve finds I - H_gg singular, with a reciprocal condition number
    # below the machine epsilon
    raise = type == 2 || (ones ? min(lo, 1.0) / max(hi, 1.0) : lo / hi) < eps()
  else
    # a cluster that alone determines part of the fit leaves no residual to estimate
    # its variance from, as a leverage of one does for HCVariance
    smin > sqrt(eps()) || return NaN
    raise = false
  end
  t = 0.0
  for q in eachindex(λ)
    s = 1 - λ[q]
    sq = raise ? max(s, 1e-12) : s
    φ = sq == s ? (type == 2 ? 1 / (sqrt(sq) * (1 + sqrt(sq))) : 1 / sq) :
                  ((type == 2 ? 1 / sqrt(sq) : 1 / sq) - 1) / λ[q]
    a = b = 0.0
    for r in eachindex(la)
      a += V[r, q] * la[r]
      b += V[r, q] * w[r]
    end
    t += φ * a * b
  end
  return t
end

# The eigenvalues λ and the eigenvectors, the columns of V, of the symmetric matrix A,
# whose upper triangle is used, by cyclic Jacobi rotations (Golub and Van Loan 2013,
# Section 8.5), in place: A is overwritten. For the small matrices of the cluster
# adjustments, this avoids the allocations of LAPACK's eigen.
function _jacobi!(λ, V, A)
  P = size(A, 1)
  for j in 1:P, i in j+1:P
    A[i, j] = A[j, i]
  end
  fill!(V, 0.0)
  for i in 1:P
    V[i, i] = 1.0
  end
  for _ in 1:30
    off = total = 0.0
    for j in 1:P, i in 1:P
      a2 = A[i, j]^2
      total += a2
      i != j && (off += a2)
    end
    off <= eps()^2 * total && break
    for p in 1:P-1, q in p+1:P
      apq = A[p, q]
      apq == 0 && continue
      θ = (A[q, q] - A[p, p]) / (2apq)
      t = (θ >= 0 ? 1.0 : -1.0) / (abs(θ) + sqrt(θ^2 + 1))
      c = 1 / sqrt(t^2 + 1)
      s = t * c
      for k in 1:P
        akp, akq = A[k, p], A[k, q]
        A[k, p], A[k, q] = c * akp - s * akq, s * akp + c * akq
      end
      for k in 1:P
        apk, aqk = A[p, k], A[q, k]
        A[p, k], A[q, k] = c * apk - s * aqk, s * apk + c * aqk
      end
      for k in 1:P
        vkp, vkq = V[k, p], V[k, q]
        V[k, p], V[k, q] = c * vkp - s * vkq, s * vkp + c * vkq
      end
    end
  end
  for i in 1:P
    λ[i] = A[i, i]
  end
  return λ, V
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
  cl = view(d.cluster, a:e)
  keys, sums = _clustersums(cl, _clusterorder(cl), ℓ, ε, Y, KY, type, compat)
  return _crscale(type, m, length(keys), P, compat) * sum(abs2, sums)
end
