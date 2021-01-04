module GettingStarted

using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
using Metalhead: CIFAR10, trainimgs
using Parameters: @with_kw
using Images: channelview
using Statistics: mean
using Base.Iterators: partition


@with_kw mutable struct Args
    batchsize::Int = 128
    throttle::Int = 10
    lr::Float64 = 1e-3
    momentum::Float64 = 0.9
    epochs::Int = 50
    splitr_::Float64 = 0.1
end

# Function to convert the RGB image to Float64 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_processed_data(args)
    # Fetching the train and validation data and getting them into proper shape	
    X = trainimgs(dataset(CIFAR10))
    imgs = [getarray(X[i].img) for i in 1:40000]   
    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)
	
    train_pop = Int((1-args.splitr_)* 40000)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, args.batchsize)]
    valset = collect(train_pop+1:40000)
    valX = cat(imgs[valset]..., dims = 4)
    valY = labels[:, valset]
	
    val = (valX,valY)
    return train, val
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(dataset(CIFAR10))

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:1000]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10)
    testX = cat(testimgs..., dims = 4)

    test = (testX,testY)
    return test
end

function net()
    return Chain(
        Conv((5, 5), 3 => 16, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 16 => 32, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(32*5*5, 256, relu),
        Dense(256, 120, relu),
        Dense(120, 10)
    )
end

accuracy(x, y, model) = mean(onecold(model(x), 1:10) .== onecold(y, 1:10))

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train, val = get_processed_data(args)

    @info("Constructing Model")
    model = net()
    loss(x, y) = logitcrossentropy(model(x), y)

    ## Training
    # Define the callback and the optimizer
    evalcb = throttle(() -> @show(loss(val...)), args.throttle)
    opt = Momentum(args.lr, args.momentum)
    @info("Training....")
    # Start to train models
    Flux.@epochs args.epochs Flux.train!(loss, params(model), train, opt; cb = evalcb)

    return model
end

function test(model)
    test_data = get_test_data()
    @show(accuracy(test_data..., model))
end

Metalhead.download(CIFAR10)
model = train()
test(model)

end # module
