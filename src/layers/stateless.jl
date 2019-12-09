# Cost/loss functions
"""
   meanabsoluteerror(ŷ, y; weight=nothing, reduction=mean)

Mean absolute error.

MAE = 1/n * sum(|y-ŷ|)

Return `1 // length(y) * sum(abs.(ŷ .- y))`
"""
function meanabsoluteerror(ŷ::AbstractArray, y::AbstractArray; weight=nothing, reduction=mean)
  # calculate mean absolute error for each sample in the batch
  mae = _meanabsoluteerror(ŷ, y, weight)
  # return mae or reduce before returning
  reduction == nothing ? mae : reduction(mae)
end

# elementwise operation
absoluteerror(ŷ::Number, y::Number) = abs(ŷ - y)

# aliasses
const ae = absoluteerror
const absolute_error = absoluteerror
const mae = meanabsoluteerror
const mean_absolute_error = meanabsoluteerror

function _meanabsoluteerror(ŷ::AbstractArray, y::AbstractArray, weight::Nothing)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* absoluteerror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _meanabsoluteerror(ŷ::AbstractArray, y::AbstractArray, weight::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* absoluteerror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _meanabsoluteerror(ŷ::AbstractArray, y::AbstractArray, weight::AbstractVector)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* absoluteerror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _meanabsoluteerror(ŷ::AbstractArray, y::AbstractArray, weight::AbstractArray)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* absoluteerror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    mse(ŷ, y)

Mean squared error.

MSE = 1/n * sum((y-ŷ)^2)

Return `1 // length(y) * sum((y .- ŷ).^2)`

# Examples
```jldoctest
julia> Flux.mse([0, 2], [1, 1])
1//1
```
"""
function meansquarederror(ŷ::AbstractArray, y::AbstractArray; weight=nothing, reduction=mean)
  # calculate mean squared error for each sample in the batch
  mse = _meansquarederror(ŷ, y, weight)
  # return mse or reduce before returning
  reduction == nothing ? mse : reduction(mse)
end

# elementwise operation
squarederror(ŷ::Number, y::Number) = (ŷ - y)^2

# aliasses
const se = squarederror
const squared_error = squarederror
const mse = meansquarederror
const mean_squared_error = meansquarederror

function _meansquarederror(ŷ::AbstractArray, y::AbstractArray, weight::Nothing)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* squarederror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _meansquarederror(ŷ::AbstractArray, y::AbstractArray, weight::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* squarederror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _meansquarederror(ŷ::AbstractArray, y::AbstractArray, weight::AbstractVector)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* squarederror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _meansquarederror(ŷ::AbstractArray, y::AbstractArray, weight::AbstractArray)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(1 // (length(ŷ)//size(ŷ)[end]) .* squarederror.(ŷ, y), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    msle(ŷ, y; ϵ=eps(eltype(ŷ)))

Return the mean of the squared logarithmic errors; calculated as
`sum((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2) / length(y)`.
The `ϵ` term provides numerical stability.

Penalizes an under-predicted estimate greater than an over-predicted estimate.
"""
msle(ŷ, y; ϵ=eps(eltype(ŷ))) = sum((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2) * 1 // length(y)

"""
    huber_loss(ŷ, y; δ=1.0)

Return the mean of the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
given the prediction `ŷ` and true values `y`.

                 | 0.5 * |ŷ - y|,            for |ŷ - y| <= δ
    Huber loss = |
                 |  δ * (|ŷ - y| - 0.5 * δ), otherwise
"""
function huber_loss(ŷ, y;  δ=eltype(ŷ)(1))
   abs_error = abs.(ŷ .- y)
   temp = abs_error .<  δ
   x = eltype(ŷ)(0.5)
   hub_loss = sum(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp)) * 1 // length(y)
end


"""
    crossentropy(ŷ, y; weight=nothing, ϵ=eps(eltype(ŷ)), reduction=mean)

Cross entropy: CE = -sum(y*log(ŷ))

Assumes that each sample in the batch is the output of a softmax function and
  thus contains probabilities between 0 and 1.

Returns the crossentropy for each sample in the batch when reduction is nothing.
Otherwise, the output is first reduced.
For example, reduction can be `sum` or `mean`.
The ϵ term provides numerical stability.

See also: [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.crossentropy(softmax([-1.1491, 0.8619, 0.3127]), [1, 1, 0])
3.085467254747739
```
"""
function crossentropy(ŷ::AbstractArray, y::AbstractArray; weight=nothing, ϵ::Number=eps(eltype(ŷ)), reduction=mean)
  # calculate crossentropy for each sample in the batch
  ce = _crossentropy(ŷ, y, weight, ϵ)
  # return ce or reduce before returning
  reduction == nothing ? ce : reduction(ce)
end

# elementwise operation
crossentropy(ŷ::Number, y::Number; ϵ::Number=eps(eltype(ŷ))) = -y * log(ŷ + ϵ)
# Re-definition to fix interaction with CuArrays
CuArrays.@cufunc crossentropy(ŷ::Number, y::Number; ϵ::Number=eps(eltype(ŷ))) = -y * log(ŷ + ϵ)

# aliasses
const categoricalcrossentropy = crossentropy
const categorical_crossentropy = crossentropy

function _crossentropy(ŷ::AbstractArray, y::AbstractArray, weight::Nothing, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(crossentropy.(ŷ, y, ϵ=ϵ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _crossentropy(ŷ::AbstractArray, y::AbstractArray, weight::Number, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(crossentropy.(ŷ, y, ϵ=ϵ), dims=(1:length(size(ŷ))-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _crossentropy(ŷ::AbstractArray, y::AbstractArray, weight::AbstractVector, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(crossentropy.(ŷ, y, ϵ=ϵ), dims=(1:length(size(ŷ))-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _crossentropy(ŷ::AbstractArray, y::AbstractArray, weight::AbstractArray, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(crossentropy.(ŷ, y, ϵ=ϵ) .* weight, dims=(1:length(size(ŷ))-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    logitcrossentropy(logŷ, y; weight=nothing)

Logit cross entropy.
Similar to cross entropy, but the inputs are logits instead of probabilities.
`logitcrossentropy(logŷ, y)` is mathematically equivalent to [`Flux.crossentropy(softmax(log(ŷ)), y)`](@ref),
but it is more numerically stable.
Only defined for vectors or matrices, as the (log)softmax can only be taken from these structures.

LCE = -1/n * sum(y*log(softmax(logŷ)))

Return `-1 // size(y, 2) * sum(y .* logsoftmax(logŷ) .* weight)`

See also: [`Flux.crossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.logitcrossentropy([-1.1491, 0.8619, 0.3127], [1, 1, 0])
3.085467254747738
```
"""
function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=nothing, reduction=mean)
  # calculate logit crossentropy for each sample in the batch
  lce = _logitcrossentropy(logŷ, y, weight)
  # return lce or reduce before returning
  reduction == nothing ? lce : reduction(lce)
end

# aliasses
const logit_crossentropy = logitcrossentropy
const logitcategoricalcrossentropy = logitcrossentropy
const logit_categorical_crossentropy = logitcrossentropy

function _logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Nothing)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(-y .* logsoftmax(logŷ), dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(-y .* logsoftmax(logŷ), dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::AbstractVector)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(-y .* logsoftmax(logŷ), dims=(1:ndims(logŷ)-1)) .* weight
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _logitcrossentropy(logŷ::AbstractMatrix, y::AbstractMatrix, weight::AbstractMatrix)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(-y .* logsoftmax(logŷ) .* weight, dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    binarycrossentropy(ŷ, y; weight=nothing, ϵ=eps(eltype(ŷ)))

Binary cross entropy / log loss.
Assumes that each sample in ŷ is the output of a sigmoid/logistic function.
Adds -log(ŷ) for y=1 and -log(1-ŷ) for y=0 to the cost for each sample.
The total cost is a weighted average of the sample costs.

BCE = -y*log(ŷ) - (1-y)*log(1-ŷ))

Return `-1 // size(y, 2) * sum((y .* log.(ŷ .+ ϵ) + (1 .- y) .* log.(1 .- ŷ .+ ϵ)) .* weight)`.
The ϵ term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.binarycrossentropy.(σ.([-1.1491, 0.8619, 0.3127]), [1, 1, 0])
3-element Array{Float64,1}:
 1.424397097347566
 0.35231664672364077
 0.8616703662235441
"""
function binarycrossentropy(ŷ::AbstractArray, y::AbstractArray; weight=nothing, ϵ::Number=eps(eltype(ŷ)), reduction=mean)
  # calculate binary crossentropy for each sample in the batch
  bce = _binarycrossentropy(ŷ, y, weight, ϵ)
  # return bce or reduce before returning
  reduction == nothing ? bce : reduction(bce)
end

# elementwise operation
binarycrossentropy(ŷ::Number, y::Number; ϵ::Number=eps(eltype(ŷ))) = -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)
# Re-definition to fix interaction with CuArrays
CuArrays.@cufunc binarycrossentropy(ŷ::Number, y::Number; ϵ::Number=eps(eltype(ŷ))) = -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ)

# aliasses
const binary_crossentropy = binarycrossentropy

function _binarycrossentropy(ŷ::AbstractArray, y::AbstractArray, weight::Nothing, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binarycrossentropy.(ŷ, y, ϵ=ϵ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _binarycrossentropy(ŷ::AbstractArray, y::AbstractArray, weight::Number, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binarycrossentropy.(ŷ, y, ϵ=ϵ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _binarycrossentropy(ŷ::AbstractArray, y::AbstractArray, weight::AbstractVector, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binarycrossentropy.(ŷ, y, ϵ=ϵ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _binarycrossentropy(ŷ::AbstractArray, y::AbstractArray, weight::AbstractArray, ϵ::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binarycrossentropy.(ŷ, y, ϵ=ϵ) .* weight, dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    logitbinarycrossentropy(logŷ, y; weight=nothing)

Logit binary cross entropy.
Similar to binary cross entropy, but the inputs are logits instead of probabilities.
`logitbinarycrossentropy(logŷ, y)` is mathematically equivalent to [`Flux.binarycrossentropy(σ(log(ŷ)), y)`](@ref),
but it is more numerically stable.
Only defined for vectors or matrices, as the (log)sigmoid can only be taken from these structures.

LBCE = -y*log(σ(logŷ)) - (1-y)*log(1-σ(logŷ))

Return `1 // size(y, 2) * sum(((1 .- y).*logŷ .- logσ.(logŷ)) .* weight)`

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref)

# Examples
```jldoctest
julia> Flux.logitbinarycrossentropy.([-1.1491, 0.8619, 0.3127], [1, 1, 0])
3-element Array{Float64,1}:
 1.4243970973475661
 0.35231664672364094
 0.8616703662235443
```
"""
function logitbinarycrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight=nothing, reduction=mean)
  # calculate logit binary crossentropy for each sample in the batch
  lbce = _logitbinarycrossentropy(logŷ, y, weight)
  # return lbce or reduce before returning
  reduction == nothing ? lbce : reduction(lbce)
end

# aliasses
const logit_binary_crossentropy = logitbinarycrossentropy

function _logitbinarycrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Nothing)
  # sum in all dimensions but the last (return one value per sample in the batch)
  # y .* logσ.(logŷ) + (1 .- y) .* log.(1 .- σ.(logŷ))
  cost_per_sample = sum((1 .- y).*logŷ .- logσ.(logŷ), dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _logitbinarycrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  # y .* logσ.(logŷ) + (1 .- y) .* log.(1 .- σ.(logŷ))
  cost_per_sample = sum((1 .- y).*logŷ .- logσ.(logŷ), dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _logitbinarycrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat, weight::AbstractVector)
  # sum in all dimensions but the last (return one value per sample in the batch)
  # y .* logσ.(logŷ) + (1 .- y) .* log.(1 .- σ.(logŷ))
  cost_per_sample = sum((1 .- y).*logŷ .- logσ.(logŷ), dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _logitbinarycrossentropy(logŷ::AbstractMatrix, y::AbstractMatrix, weight::AbstractMatrix)
  # sum in all dimensions but the last (return one value per sample in the batch)
  # y .* logσ.(logŷ) + (1 .- y) .* log.(1 .- σ.(logŷ))
  cost_per_sample = sum(((1 .- y).*logŷ .- logσ.(logŷ)) .* weight, dims=(1:ndims(logŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    focalloss(ŷ, y; γ=2., weight=nothing)

Focal loss.
Assumes that each sample in ŷ is the output of a softmax function.
Adds -y*log(ŷ) to the cost for each sample.
The total cost is a weighted average of the sample costs.

     C
FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
    c=1

where C = number of classes, c = class and o = observation
Is already numerically stable, because it calls crossentropy with ϵ=eps(eltype(ŷ)).
ATTENTION: Alpha is by default 0.25 in the original papper, but here the weight must be explicitly specified!
"""
function focalloss(ŷ::AbstractArray, y::AbstractArray; γ::Number=2., weight=nothing, reduction=mean)
  # calculate focal loss for each sample in the batch
  fl = _focalloss(ŷ, y, γ, weight)
  # return fl or reduce before returning
  reduction == nothing ? fl : reduction(fl)
end

# elementwise operation
focalloss(ŷ::Number, y::Number; γ::Number=2.) = (1 - ŷ)^γ * crossentropy(ŷ, y)
# Re-definition to fix interaction with CuArrays
CuArrays.@cufunc focalloss(ŷ::Number, y::Number; γ::Number=2.) = (1 - ŷ)^γ * crossentropy(ŷ, y)

# aliasses
const focal_loss = focalloss
const categoricalfocalloss = focalloss
const categorical_focal_loss = focalloss

function _focalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::Nothing)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(focalloss.(ŷ, y, γ=γ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _focalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(focalloss.(ŷ, y, γ=γ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _focalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::AbstractVector)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(focalloss.(ŷ, y, γ=γ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _focalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::AbstractArray)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(focalloss.(ŷ, y, γ=γ) .* weight, dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    logitfocalloss(logŷ, y; γ=2., α=0.25)

Logit focal loss.
ATTENTION: Alpha is by default 0.25 in the original papper, but here the weight must be explicitly specified!
"""
logitfocalloss(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; γ::Number=2., weight=nothing, reduction=mean) = focalloss(softmax(logŷ), y, γ=γ, weight=weight, reduction=reduction)

# aliasses
const logit_focal_loss = logitfocalloss
const logitcategoricalfocalloss = logitfocalloss
const logit_categorical_focal_loss = logitfocalloss

"""
    binaryfocalloss(ŷ, y; γ=2., weight=nothing, reduction=mean)

Binary focal loss.
FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
Is already numerically stable, because it calls crossentropy with ϵ=eps(eltype(ŷ)).
ATTENTION: Alpha is by default 0.25 in the original papper, but here the weight must be explicitly specified!
"""
function binaryfocalloss(ŷ::AbstractArray, y::AbstractArray; γ::Number=2., weight=nothing, reduction=mean)
  # calculate binary focal loss for each sample in the batch
  bfl = _binaryfocalloss(ŷ, y, γ, weight)
  # return bfl or reduce before returning
  reduction == nothing ? bfl : reduction(bfl)
end

# elementwise operation
binaryfocalloss(ŷ::Number, y::Number; γ::Number=2.) = (1 - ŷ)^γ * crossentropy(ŷ, y) + ŷ^γ * crossentropy((1-ŷ), (1-y))
# Re-definition to fix interaction with CuArrays
CuArrays.@cufunc binaryfocalloss(ŷ::Number, y::Number; γ::Number=2.) = (1 - ŷ)^γ * crossentropy(ŷ, y) + ŷ^γ * crossentropy((1-ŷ), (1-y))

# aliasses
const binary_focal_loss = binaryfocalloss

function _binaryfocalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::Nothing)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binaryfocalloss.(ŷ, y, γ=γ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

function _binaryfocalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::Number)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binaryfocalloss.(ŷ, y, γ=γ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) * weight
end

function _binaryfocalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::AbstractVector)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binaryfocalloss.(ŷ, y, γ=γ), dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1)) .* weight
end

function _binaryfocalloss(ŷ::AbstractArray, y::AbstractArray, γ::Number, weight::AbstractArray)
  # sum in all dimensions but the last (return one value per sample in the batch)
  cost_per_sample = sum(binaryfocalloss.(ŷ, y, γ=γ) .* weight, dims=(1:ndims(ŷ)-1))
  # squeeze
  return dropdims(cost_per_sample, dims=Dims(1:ndims(cost_per_sample)-1))
end

"""
    logitbinaryfocalloss(logŷ, y; γ=2., α=0.25)

Logit binary focal loss.
ATTENTION: Alpha is by default 0.25 in the original papper, but here the weight must be explicitly specified!
"""
logitbinaryfocalloss(logŷ::AbstractVecOrMat, y::AbstractVecOrMat; γ::Number=2., weight=nothing, reduction=mean) = binaryfocalloss(σ.(logŷ), y, γ=γ, weight=weight, reduction=reduction)

# aliasses
const logit_binary_focal_loss = logitbinaryfocalloss

"""
    kldivergence(ŷ, y)

Return the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given probability distributions.

KL divergence is a measure of how much one probability distribution is different
from the other.
It is always non-negative and zero only when both the distributions are equal
everywhere.
"""
function kldivergence(ŷ, y)
  entropy = sum(y .* log.(y)) * 1 //size(y,2)
  cross_entropy = crossentropy(ŷ, y)
  return entropy + cross_entropy
end

"""
    poisson(ŷ, y)

Return how much the predicted distribution `ŷ` diverges from the expected Poisson
distribution `y`; calculated as `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`.

[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson(ŷ, y) = sum(ŷ .- y .* log.(ŷ)) * 1 // size(y,2)

"""
    hinge(ŷ, y)

Return the [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as
`sum(max.(0, 1 .- ŷ .* y)) / size(y, 2)`.

See also: [`squared_hinge`](@ref)
"""
hinge(ŷ, y) = sum(max.(0, 1 .-  ŷ .* y)) * 1 // size(y, 2)

"""
    squared_hinge(ŷ, y)

Return the squared hinge loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as `sum((max.(0, 1 .- ŷ .* y)).^2) / size(y, 2)`.

See also: [`hinge`](@ref)
"""
squared_hinge(ŷ, y) = sum((max.(0, 1 .- ŷ .* y)).^2) * 1 // size(y, 2)

"""
    dice_coeff_loss(ŷ, y; smooth=1)

Return a loss based on the dice coefficient.
Used in the [V-Net](https://arxiv.org/pdf/1606.04797v1.pdf) image segmentation
architecture.
Similar to the F1_score. Calculated as:
    1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)`
"""
dice_coeff_loss(ŷ, y; smooth=eltype(ŷ)(1.0)) = 1 - (2*sum(y .* ŷ) + smooth) / (sum(y.^2) + sum(ŷ.^2) + smooth)

"""
    tversky_loss(ŷ, y; β=0.7)

Return the [Tversky loss](https://arxiv.org/pdf/1706.05721.pdf).
Used with imbalanced data to give more weight to false negatives.
Larger β weigh recall higher than precision (by placing more emphasis on false negatives)
Calculated as:
    1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)
"""
tversky_loss(ŷ, y; β=eltype(ŷ)(0.7)) = 1 - (sum(y .* ŷ) + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)

# Help functions
"""
    normalise(x; dims=1)

Normalise `x` to mean 0 and standard deviation 1 across the dimensions given by `dims`.
Defaults to normalising over columns.

```jldoctest
julia> a = reshape(collect(1:9), 3, 3)
3×3 Array{Int64,2}:
 1  4  7
 2  5  8
 3  6  9

julia> Flux.normalise(a)
3×3 Array{Float64,2}:
 -1.22474  -1.22474  -1.22474
  0.0       0.0       0.0
  1.22474   1.22474   1.22474

julia> Flux.normalise(a, dims=2)
3×3 Array{Float64,2}:
 -1.22474  0.0  1.22474
 -1.22474  0.0  1.22474
 -1.22474  0.0  1.22474
```
"""
function normalise(x::AbstractArray; dims=1)
  μ′ = mean(x, dims = dims)
  σ′ = std(x, dims = dims, mean = μ′, corrected=false)
  return (x .- μ′) ./ σ′
end

"""
    flatten(x::AbstractArray)

Transform (w, h, c, b)-shaped input into (w × h × c, b)-shaped output
by linearizing all values for each element in the batch.
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end
