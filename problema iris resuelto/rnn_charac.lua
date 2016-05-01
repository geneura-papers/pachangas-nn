
require "nn"
require "parser"

opt = {
    data_dir = '.',

    -- model Parameters

    inputs = 2;
    outputs = 1;
    HUs = 20;

    -- optimization parameters

    learningRate = 0.01,        -- : This is the learning rate used during training. The update of the parameters will be parameters = parameters - learningRate * parameters_gradient. Default value is 0.01.
    learningRateDecay = 0.97,      -- : The learning rate decay. If non-zero, the learning rate (note: the field learningRate will not change value) will be computed after each iteration (pass over the dataset) with: current_learning_rate =learningRate / (1 + iteration * learningRateDecay)
    maxIteration = 30,          -- : The maximum number of iteration (passes over the dataset). Default is 25.
    shuffleIndices = True,      -- : Boolean which says if the examples will be randomly sampled or not. Default is true. If false, the examples will be taken in the order of the dataset.
    hookExample = False,         -- : A possible hook function which will be called (if non-nil) during training after each example forwarded and backwarded through the network. The function takes (self, example) as parameters. Default is nil.
    hookIteration = nil,         -- : A possible hook function which will be called (if non-nil) during training after a complete pass over the dataset. The function takes (self, iteration, currentError) as parameters. Default is nil.




    -- bookkeeping


    seed = 123,                         -- torch manual random number generator seed
    print_every = 1,                    -- how many steps/minibatches between printing out the loss
    eval_val_every = 1000,              -- every how many iterations should we evaluate on validation data?
    checkpoint_dir = 'checkpoint',      -- output directory where checkpoints get written
    savefile = 'mlp',                   -- filename to autosave the checkpont to. Will be inside checkpoint_dir/






    -- GPU/CPU

    gpuid = -1,                         -- which gpu to use. -1 = use CPU
}


--MLP


-- Creating the RNA

mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 4; outputs = 1; HUs = 20; -- parameters

mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

--Training

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = opt.learningRate
trainer.learningRateDecay = opt.learningRateDecay
trainer.maxIteration = opt.maxIteration
trainer.shuffleIndices = opt.shuffleIndices
trainer.hookExample = opt.hookExample
trainer.hookIteration = opt.hookIteration
trainer:train(dataset)

--5.1,3.5,1.4,0.2,Iris-setosa

x = torch.Tensor(4)
x[1] = 5; x[2] = 3.4; x[3] = 1.6 ; x[4] = 0.15; print(mlp:forward(x))
