A project to analyze reinforcement learning algorithms on the Atari game Pong.

Based off of Andrej Karpathy's [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/), but with extensions to support training multiple models concurrently.

The model used is simply an MLP, but with enough training (~5,000 games) will be on par with the built-in AI.

The purpose of the project is to evaluate the dependency on the hyperparameters and to get a feel for training RL agents.  The analysis is recorded in the notebook [eval](https://github.com/maitchison/pong/blob/master/PONG%20eval.ipynb).

Important takeaways are as follows

* The model trains well even with a small number of hidden units (25).  Even 2 hidden units show promise if training is left for long enough.

* A batch size of 1 causes the model to not train properly, all other batch sizes seem fine.

* Weight decay has little effect on training

* An appropriate discount rate is required as if it is too low the model will not be able to match the actions it performed with scoring a point.  In this case, a value of .95 or higher is sufficient.

## Future extensions
Move to TensorFlow or PyTorch
Switch to a convolutional neural net, perhaps with LSTM


