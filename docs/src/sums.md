# [Kernel sums](@id sums-page)

All estimators of the package are built from the sums

```math
S_{rj} = \sum_{i=1}^n w_i z_i\, K(u_{ij})\, u_{ij}^r, \qquad u_{ij} = \frac{x_i - p_j}{h_j}, \qquad r = 0, \dots, k_{\max},
```

over the sorted observations ``x_i`` with weights ``w_i`` and values ``z_i``, at the
evaluation points ``p_j`` with bandwidths ``h_j``. For a symmetric kernel,
``S_{0j} / (h_j \sum_i w_i)`` is the kernel density estimate at ``p_j``, and with
``k_{\max} = 2p``, and with ``z = y`` and ``k_{\max} = p``, the sums give the normal
equations of a local polynomial fit of order ``p``. [`kernelsums`](@ref) returns
the ``(k_{\max} + 1) \times m`` matrix of these sums, with one bandwidth or one per
point, and computes them with one of four engines.

| engine | kernels | cost | accuracy |
|---|---|---|---|
| [`DirectSums`](@ref) | any | ``O(nm)`` | exact; the reference |
| [`WindowSums`](@ref) | compactly supported | ``O(m \log n)`` plus the total window size | identical to `DirectSums` |
| [`FastSums`](@ref) | polynomial | ``O(n(D + k_{\max}) + m \log m)`` | about ``10^{-13}`` of the nearby weight |
| [`BinnedSums`](@ref) | any | ``O(n)`` plus windows over at most ``2n`` grid points | of order ``\mathrm{binsperh}^{-2}`` for continuous kernels |

Here ``D`` is the degree of the pieces of the kernel. By default the package uses
`WindowSums` for a polynomial kernel when the windows hold few observations,
`FastSums` otherwise, and `DirectSums` for other kernels.

`FastSums` uses that on each polynomial piece of the kernel, ``S_{rj}`` is a linear
combination of power sums of the observations in that piece, which prefix sums
give in constant time (Seifert, Brockmann, Engel and Gasser 1994; Langrené and Warin
2019). Power sums about one centre cancel catastrophically when the data are far
from zero or the bandwidth is small compared with their range. Therefore, the
engine cuts the data into blocks one window wide and stores the prefix sums of
each block about its own centre, so that its rounding error stays at about
``10^{-13}`` of the weight near the point, whatever the location and the scale of
the data. `BinnedSums` splits each value between the two nearest points of a grid
with spacing ``h/\mathrm{binsperh}`` and sums exactly over the occupied grid
points, which suits Gaussian kernels with many observations.

The points are processed in chunks as parallel tasks when Julia runs with several
threads, and the sums do not depend on the number of threads. A
[`SumsWorkspace`](@ref) holds buffers that repeated calls, such as those of a
bandwidth search, reuse.

```@autodocs
Modules = [KernelSmoothing]
Pages = ["engines.jl"]
```
