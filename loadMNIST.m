function [ data, labels ] = loadMNIST()
%% This code loads MNIST dataset
%% data: actual images
%% labels: classfication index

data   = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

end
