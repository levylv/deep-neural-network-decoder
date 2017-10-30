# Performance Evaluation of Channel Decoding With Deep Neural Networks 
Paper about the deep neural network decoder.

## Introduction

We propose three types of NND, which build upon multi-layer perceptron (MLP), convolution neural network (CNN) and recurrent neural network (RNN).
## Steps of Code

1. Generate the training samples saved at `data/` under `RNN/`.

```
cd RNN/
python get_data.py
```

2. Run the deep neural network, BER is saved at `result/`.

```
python mlp.py
python cnn.py
python lstm.py
```

3. Plot the figure.
```
python plot.py
```

