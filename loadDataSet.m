function [ trainData, trainLabels, testData, testLabels ] = loadDataSet( )
% Loads benchmark dataset

global BENCHMARK;

trainData   = [];   trainLabels = [];
testData    = [];   testLabels  = [];

if strcmp(BENCHMARK, 'CIFAR10')

    tempData    = load('cifar-10/data_batch_1.mat');
    trainData   = [trainData; double(tempData.data)/255];
    trainLabels = [trainLabels; tempData.labels];
    
    tempData    = load('cifar-10/data_batch_2.mat');
    trainData   = [trainData; double(tempData.data)/255];
    trainLabels = [trainLabels; tempData.labels];
    
    tempData    = load('cifar-10/data_batch_3.mat');
    trainData   = [trainData; double(tempData.data)/255];
    trainLabels = [trainLabels; tempData.labels];
    
    tempData    = load('cifar-10/data_batch_4.mat');
    trainData   = [trainData; double(tempData.data)/255];
    trainLabels = [trainLabels; tempData.labels];
    
    tempData    = load('cifar-10/data_batch_5.mat');
    trainData   = [trainData; double(tempData.data)/255];
    trainLabels = [trainLabels; tempData.labels];
    
    tempData    = load('cifar-10/test_batch.mat');
    testData    = [testData; double(tempData.data)/255];
    testLabels  = [testLabels; tempData.labels];
    
elseif strcmp(BENCHMARK, 'LETTER')
    % A: 0, Z: 25
    % train data: 1~16000 (16000), test data: 16001~20000 (4000)
    tempFile   = importdata('letter.xlsx');
    inData     = [int8(cell2mat(tempFile.textdata))-int8('A'), tempFile.data];
    
    num_train  = 16000;
    trainData  = double(inData(1:num_train,2:end))';
    trainLabels = inData(1:num_train,1)';
    
    testData   = double(inData(num_train+1:end,2:end))';
    testLabels = inData(num_train+1:end,1)';
    
elseif strcmp(BENCHMARK, 'SPAM')
    % 4601 (1813 Spam)
    % train data: 4101 (2488: non-spam, 1613: spam), test data: 500 (300: non-spam, 200:spam)
    tempFile    = load('spam/spambase.data');
    
    num_train   = 4101;
    trainData   = tempFile(201:200+num_train,1:end-1)';
    trainLabels = tempFile(201:200+num_train,end)';
    
    testData    = [tempFile(1:200,1:end-1); tempFile(end-299:end,1:end-1)]';
    testLabels  = [tempFile(1:200,end); tempFile(end-299:end,end)]';
    
end


end

