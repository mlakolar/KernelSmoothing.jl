# Reference values from RDHonest 1.0.2; recreate with rdhonest_mrot_golden.R.
# Discovery run 20261025-ks-mrot77 (R 4.5.3 container). M = nothing where RDHonest stops
# with "Insufficient unique values of the running variable to compute rule of thumb for M".
const MROT_FEW = [
  (side = :left, k = 1, tied = true, M = nothing),
  (side = :left, k = 1, tied = false, M = nothing),
  (side = :left, k = 2, tied = true, M = nothing),
  (side = :left, k = 2, tied = false, M = nothing),
  (side = :left, k = 3, tied = true, M = nothing),
  (side = :left, k = 3, tied = false, M = nothing),
  (side = :left, k = 4, tied = true, M = nothing),
  (side = :left, k = 4, tied = false, M = nothing),
  (side = :left, k = 5, tied = true, M = 3.8523458348024677),
  (side = :left, k = 5, tied = false, M = 19.700075805562619),
  (side = :left, k = 6, tied = true, M = 28.871941127016477),
  (side = :left, k = 6, tied = false, M = 29.943772142525177),
  (side = :right, k = 1, tied = true, M = nothing),
  (side = :right, k = 1, tied = false, M = nothing),
  (side = :right, k = 2, tied = true, M = nothing),
  (side = :right, k = 2, tied = false, M = nothing),
  (side = :right, k = 3, tied = true, M = nothing),
  (side = :right, k = 3, tied = false, M = nothing),
  (side = :right, k = 4, tied = true, M = nothing),
  (side = :right, k = 4, tied = false, M = nothing),
  (side = :right, k = 5, tied = true, M = 6.5624236558643911),
  (side = :right, k = 5, tied = false, M = 15.495396536204083),
  (side = :right, k = 6, tied = true, M = 23.03186862873477),
  (side = :right, k = 6, tied = false, M = 30.000876112384738),
]
