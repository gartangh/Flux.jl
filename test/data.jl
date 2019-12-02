using Flux.Data
using Test

@test cmudict()["CATASTROPHE"] == :[K,AH0,T,AE1,S,T,R,AH0,F,IY0].args

@test length(CMUDict.phones()) == 39

@test length(CMUDict.symbols()) == 84

@test MNIST.images()[1] isa Matrix
@test MNIST.labels() isa Vector{Int64}

@test FashionMNIST.images()[1] isa Matrix
@test FashionMNIST.labels() isa Vector{Int64}

@test Data.Sentiment.train() isa Vector{Data.Tree{Any}}

@test Iris.features() isa Matrix
@test size(Iris.features()) == (4,150)

@test Iris.labels() isa Vector{String}
@test size(Iris.labels()) == (150,)

@test size(COCO.test_images()) = (10,)
@test size(COCO.train_images()) = (100,)
@test size(COCO.val_images()) = (20,)
@test size(COCO.test_labels()) = (10)
@test size(COCO.train_labels()) = (100,)
@test size(COCO.val_labels()) = (20,)
