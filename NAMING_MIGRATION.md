# Naming migration

Versions and UUIDs are unchanged. Workspace dependencies use local `master`
checkouts; see `CONTRIBUTING.md` for the naming policy.

| Previous public name | Canonical name |
| --- | --- |
| `kernelsums`, `kernelsums!` | `kernel_sums`, `kernel_sums!` |
| `selectbandwidth` | `select_bandwidth` |
| `weightsum` | `weight_sum` |
| `worstcasebias` | `worst_case_bias` |
| `localpolynomial` | `local_polynomial` |
| `smoothermatrix` | `smoother_matrix` |
| `confband` | `conf_band` |
| `Honest.biascorrected`, keyword `biascorrected` | `bias_corrected` |

Deprecated function names are bindings to the same generic functions, so
existing downstream method definitions remain visible through canonical names.
The legacy keyword and property remain accepted; supplying both keyword
spellings throws. Numerical defaults, positional constructors, mathematical
identifiers and external StatsAPI/Distributions interface names are unchanged.
