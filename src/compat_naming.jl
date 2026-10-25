# Deprecated names remain bindings to the same generic, including downstream methods.
Base.@deprecate_binding kernelsums kernel_sums
Base.@deprecate_binding kernelsums! kernel_sums!
Base.@deprecate_binding selectbandwidth select_bandwidth
Base.@deprecate_binding weightsum weight_sum
Base.@deprecate_binding worstcasebias worst_case_bias
Base.@deprecate_binding localpolynomial local_polynomial
Base.@deprecate_binding smoothermatrix smoother_matrix
Base.@deprecate_binding confband conf_band

function Base.getproperty(m::Honest, name::Symbol)
  if name === :biascorrected
    Base.depwarn("Honest.biascorrected is deprecated; use bias_corrected", :biascorrected)
    return getfield(m, :bias_corrected)
  end
  return getfield(m, name)
end
Base.propertynames(::Honest, private::Bool = false) = (fieldnames(Honest)..., :biascorrected)
