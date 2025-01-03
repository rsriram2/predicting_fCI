net = alexnet;
layers = net.Layers;
layers = layers(1: end - 3);
layers(end + 1) = fullyConnectedLayer(1, 'Name', 'special_2');
layers(end + 1) = reluLayer;
layers(end + 1) = regressionLayer('Name', 'output');

testData = {concatenatedTestFiles, concatenatedTestLabels};

% sizeImage = net.Layers(1).InputSize;
% aTrainingSet = augmentedImageDatastore(sizeImage, trainIMDS, 'ColorPreprocessing', 'gray2rgb');
% aTestSet = augmentedImageDatastore(sizeImage, testIMDS, 'ColorPreprocessing', 'gray2rgb');
% aValSet = augmentedImageDatastore(sizeImage, valIMDS, 'ColorPreprocessing', 'gray2rgb');
    
opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none', ...
    'Verbose', true, ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 256, ...
    'Plots', 'training-progress', 'ValidationData', testData, 'ValidationFrequency', 100);

[model, traininfo] = trainNetwork(shuffledTrain2Files, shuffledTrain2Labels, model.Layers, opts);

prediction = predict(model, resizedTestFiles);

valLabels = valIMDS.Labels;
matrix = confusionmat(testLabels, prediction);
display(matrix);