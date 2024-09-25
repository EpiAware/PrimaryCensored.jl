@doc raw"
A struct that contains the two different distributions, d1 and d2

"



struct PrimaryCensoredBeta{T<:Distributions.VariateForm,
    S<:Distributions.ValueSupport} <:Distributions.Sampleable{T, S}
    d1::Distributions.Sampleable{T,S}
    d2::Distributions.Sampleable{T,S}
end

#! TODO clean up outside constructor
PrimaryCensored(
                d1::Distributions.Sampleable{S, T},
                d2::Distributions.Sampleable{S, T}) where 
                {S<:Distributions.VariateForm, T<:Distributions.ValueSupport} = 
                PrimaryCensored{S,T}(d1,d2)

# Method for single value
function Base.rand(pc::PrimaryCensored)
    rand(pc.d1) .+ rand(pc.d2)
end

# Method for multiple samples
function Base.rand(pc::PrimaryCensored, n::Int)
    d1_samps = rand(pc.d1, n)
    d2_samps = rand(pc.d2, n)
    return d1_samps + d2_samps
end
                PrimaryCensoredBeta{S,T}(d1,d2)

