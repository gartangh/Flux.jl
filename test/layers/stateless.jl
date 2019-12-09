using Test
using Distributions
using Flux: onehotbatch, σ, mean, std
            absoluteerror, meanabsoluteerror, squarederror, meansquarederror,
            crossentropy, logitcrossentropy, binarycrossentropy, logitbinarycrossentropy,
            focalloss, logitfocalloss, binaryfocalloss, logitbinaryfocalloss,
            flatten

const ϵ = 1e-7

@testset "Losses" begin
  ŷ = [0.9, 0.1, 0.1, 0.9]
  logŷ = [1., 2., 3., 4.]
  y = [1, 0, 1, 0]
  ẑ = rand(Uniform(0.0,1.0),(2,2,2,4,16))
  z = randn(2,2,2,4,16)

  @testset "meanabsoluteerror" begin
    cost = [0.1, 0.1, 0.9, 0.9]
    @test absoluteerror.(ŷ, y) ≈ cost
    @test meanabsoluteerror(ŷ, y, weight=nothing, reduction=nothing) ≈ cost
    @test meanabsoluteerror(ŷ, y, weight=0.5, reduction=nothing) ≈ cost / 2
    @test meanabsoluteerror(ŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1]
    @test size(meanabsoluteerror(ẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    @test size(meanabsoluteerror(ẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test meanabsoluteerror(ŷ, y, reduction=sum) == sum(cost)
    @test meanabsoluteerror(ŷ, y, reduction=mean) == mean(cost)
  end

  @testset "mae" begin
    @test Flux.mae(ŷ, y) ≈ 1/2
  end

  @testset "huber_loss" begin
    @test Flux.huber_loss(ŷ, y) ≈ 0.20500000000000002
  end

  y = [123.0,456.0,789.0]
  ŷ = [345.0,332.0,789.0]
  @testset "msle" begin
    @test Flux.msle(ŷ, y) ≈ 0.38813985859136585
  end

  @testset "meansquarederror" begin
    cost = [0.01, 0.01, 0.81, 0.81]
    @test squarederror.(ŷ, y) ≈ cost
    @test meansquarederror(ŷ, y, weight=nothing, reduction=nothing) ≈ cost
    @test meansquarederror(ŷ, y, weight=0.5, reduction=nothing) ≈ cost / 2
    @test meansquarederror(ŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1]
    @test size(meansquarederror(ẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    @test size(meansquarederror(ẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test meansquarederror(ŷ, y, reduction=sum) == sum(cost)
    @test meansquarederror(ŷ, y, reduction=mean) == mean(cost)
  end

  @testset "crossentropy" begin
    cost = [0.10536051565782603, 0.0, 2.3025850929940432, 0.0]
    @test crossentropy.(ŷ, y) ≈ cost
    @test crossentropy(ŷ, y, weight=nothing, reduction=nothing) ≈ cost
    @test crossentropy(ŷ, y, weight=0.5, reduction=nothing) ≈ cost / 2
    @test crossentropy(ŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1]
    @test size(crossentropy(ẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    @test size(crossentropy(ẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test crossentropy(ŷ, y, reduction=sum) == sum(cost)
    @test crossentropy(ŷ, y, reduction=mean) == mean(cost)
  end

  @testset "logitcrossentropy" begin
    cost = crossentropy(softmax(logŷ), y, reduction=nothing)
    # @test logitcrossentropy.(ŷ, y) ≈ cost
    @test logitcrossentropy(logŷ, y, weight=nothing, reduction=nothing) ≈ cost
    @test logitcrossentropy(logŷ, y, weight=0.5, reduction=nothing) ≈ cost / 2
    @test logitcrossentropy(logŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1]
    # @test size(logitcrossentropy(logẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    # @test size(logitcrossentropy(logẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test logitcrossentropy(logŷ, y, reduction=sum) ≈ sum(cost)
    @test logitcrossentropy(logŷ, y, reduction=mean) ≈ mean(cost)
  end

  @testset "binarycrossentropy" begin
    cost = [0.10536051565782603, 0.10536051565782603, 2.3025850929940432, 2.3025850929940432]
    @test binarycrossentropy.(ŷ, y) ≈ cost
    @test binarycrossentropy(ŷ, y, weight=nothing, reduction=nothing) ≈ cost
    @test binarycrossentropy(ŷ, y, weight=0.5, reduction=nothing) ≈ cost / 2
    @test binarycrossentropy(ŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1]
    @test size(binarycrossentropy(ẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    @test size(binarycrossentropy(ẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test binarycrossentropy(ŷ, y, reduction=sum) ≈ sum(cost)
    @test binarycrossentropy(ŷ, y, reduction=mean) ≈ mean(cost)
  end

  @testset "logitbinarycrossentropy" begin
    cost = binarycrossentropy(σ.(logŷ), y, reduction=nothing)
    # @test logitbinarycrossentropy.(ŷ, y) ≈ cost
    @test logitbinarycrossentropy(logŷ, y, weight=nothing, reduction=nothing) ≈ cost
    @test logitbinarycrossentropy(logŷ, y, weight=0.5, reduction=nothing) ≈ cost / 2
    @test logitbinarycrossentropy(logŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1]
    # @test size(logitbinarycrossentropy(logẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    # @test size(logitbinarycrossentropy(logẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test logitbinarycrossentropy(logŷ, y, reduction=sum) ≈ sum(cost)
    @test logitbinarycrossentropy(logŷ, y, reduction=mean) ≈ mean(cost)
  end

  @testset "focalloss" begin
    cost = [0.0002634012891445649, 0.0, 0.46627348133129376, 0.0]
    @test focalloss.(ŷ, y) ≈ cost * 4
    @test focalloss(ŷ, y, weight=nothing, reduction=nothing) ≈ cost * 4
    @test focalloss(ŷ, y, weight=0.25, reduction=nothing) ≈ cost
    @test focalloss(ŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1] * 4
    @test size(focalloss(ẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    @test size(focalloss(ẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test focalloss(ŷ, y, weight=0.25, reduction=sum) == sum(cost)
    @test focalloss(ŷ, y, weight=0.25, reduction=mean) == mean(cost)
  end

  @testset "binaryfocalloss" begin
    cost = [0.0002634012891445649, 0.0002634012891445649, 0.46627348133129376, 0.46627348133129376]
    @test binaryfocalloss.(ŷ, y) ≈ cost * 4
    @test binaryfocalloss(ŷ, y, weight=nothing, reduction=nothing) ≈ cost * 4
    @test binaryfocalloss(ŷ, y, weight=0.25, reduction=nothing) ≈ cost
    @test binaryfocalloss(ŷ, y, weight=[-1, 1, -1, 1], reduction=nothing) ≈ cost .* [-1, 1, -1, 1] * 4
    @test size(binaryfocalloss(ẑ, z, weight=randn(16), reduction=nothing)) == (16,)
    @test size(binaryfocalloss(ẑ, z, weight=randn(2,2,2,4,16), reduction=nothing)) == (16,)
    @test binaryfocalloss(ŷ, y, weight=0.25, reduction=sum) ≈ sum(cost)
    @test binaryfocalloss(ŷ, y, weight=0.25, reduction=mean) ≈ mean(cost)
  end

  y = [1 2 3]
  ŷ = [4.0 5.0 6.0]
  @testset "kldivergence" begin
    @test Flux.kldivergence(ŷ, y) ≈ -1.7661057888493457
    @test Flux.kldivergence(y, y) ≈ 0
  end

  y = [1 2 3 4]
  ŷ = [5.0 6.0 7.0 8.0]
  @testset "hinge" begin
    @test Flux.hinge(ŷ, y) ≈ 0
    @test Flux.hinge(y, 0.5 .* y) ≈ 0.125
  end

  @testset "squared_hinge" begin
    @test Flux.squared_hinge(ŷ, y) ≈ 0
    @test Flux.squared_hinge(y, 0.5 .* y) ≈ 0.0625
  end

  y = [0.1 0.2 0.3]
  ŷ = [0.4 0.5 0.6]
  @testset "poisson" begin
    @test Flux.poisson(ŷ, y) ≈ 0.6278353988097339
    @test Flux.poisson(y, y) ≈ 0.5044459776946685
  end

  y = [1.0 0.5 0.3 2.4]
  ŷ = [0 1.4 0.5 1.2]
  @testset "dice_coeff_loss" begin
    @test Flux.dice_coeff_loss(ŷ, y) ≈ 0.2799999999999999
    @test Flux.dice_coeff_loss(y, y) ≈ 0.0
  end

  @testset "tversky_loss" begin
    @test Flux.tversky_loss(ŷ, y) ≈ -0.06772009029345383
    @test Flux.tversky_loss(ŷ, y, β = 0.8) ≈ -0.09490740740740744
    @test Flux.tversky_loss(y, y) ≈ -0.5576923076923075
  end

  @testset "no spurious promotions" begin
    for T in (Float32, Float64)
      y = rand(T, 2)
      ŷ = rand(T, 2)
      for f in (meanabsoluteerror, meansquarederror, Flux.huber_loss, Flux.msle,
              crossentropy, logitcrossentropy, Flux.kldivergence, Flux.hinge, Flux.poisson,
              focalloss, logitfocalloss, binaryfocalloss, logitbinaryfocalloss,
              Flux.squared_hinge, Flux.dice_coeff_loss, Flux.tversky_loss)
        fwd, back = Flux.pullback(f, ŷ, y)
        @test fwd isa T
        @test eltype(back(one(T))[1]) == T
      end
    end
end

@testset "helpers" begin
  @testset "flatten" begin
    x = randn(Float32, 10, 10, 3, 2)
    @test size(flatten(x)) == (300, 2)
  end
end
