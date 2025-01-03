net = resnet50;
lgraph = layerGraph(net);

lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

numOutputs = 1; 
newLayers = [
    fullyConnectedLayer(numOutputs, 'Name', 'fc_regression')
    regressionLayer('Name', 'regressionoutput')];

lgraph = addLayers(lgraph, newLayers);

lgraph = connectLayers(lgraph, 'avg_pool', 'fc_regression');


%testData = {resizedTestFiles, concatenatedTestLabels};

opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none', ...
    'Verbose', true, ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 16, ...
    'Plots', 'training-progress', 'ExecutionEnvironment', 'parallel');

[model, traininfo] = traiNetwork(resizedTestFiles(:,:,:,1:100), concatenatedTestLabels(1:100), lgraph, opts);

prediction = predict(model, resizedTestFiles);

%cams
featureMaps = activations(model, resizedTestFiles(:,:,:,9001), 'fc_regression');