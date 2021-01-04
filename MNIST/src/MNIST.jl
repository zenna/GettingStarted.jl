module MNIST

using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy
using Flux: @epochs
using Statistics
using MLDatasets

function run()
    x_train, y_train = MLDatasets.MNIST.traindata(Float32)
    x_test, y_test = MLDatasets.MNIST.testdata(Float32)

    # flatten images 
    x_train = Flux.flatten(x_train)
    x_test = Flux.flatten(x_test)

    # encode labels
    y_train = onehotbatch(y_train, 0:9)
    y_test = onehotbatch(y_test, 0:9)

    # create the full dataset
    train_data = DataLoader(x_train, y_train, batchsize=128)
    test_data = DataLoader(x_test, y_test, batchsize=128)

    model = Chain(
        Dense(prod((28,28,1)), 32, relu),
        Dense(32, 10),
        softmax)

    function accuracy(y_pred, y)
        mean(onecold(y_pred) .== onecold(y))
    end

    function loss(x, y)
        crossentropy(model(x), y)
    end

    lr = 0.1
    opt = Descent(lr)
    ps = Flux.params(model)

    number_epochs = 10
    @epochs number_epochs Flux.train!(loss, ps, train_data, opt)
    @show accuracy(model(x_train), y_train)
    @show accuracy(model(x_test), y_test)
end # run()

end # module
