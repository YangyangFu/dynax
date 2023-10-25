# 2023/10/24
- [] Druing training, the neural ODE for dynamcial system just predict one step or the steps based on sample data? 
  - The difficulty of multi-step prediction is that the neural ODE will act like a RNN, rolling out the prediction step by step. If the steps are too long, the error will accumulate and the prediction will be bad. For training, there might be gradient vanishing problem.  
  - check this in a test script: long-horizon training (extreme case is one sample with the whole trajectory) -> very hard to train with long sequences, error propagation leads to prediction explosion, and possible `NANs` in grads.
