trainMatFiles = dir(fullfile('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Train/', '*.mat'));

net = resnet50;
lgraph = layerGraph(net);

lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

numOutputs = 1; 
newLayers = [
    fullyConnectedLayer(numOutputs, 'Name', 'fc_regression')
    regressionLayer('Name', 'regressionoutput')];

lgraph = addLayers(lgraph, newLayers);

lgraph = connectLayers(lgraph, 'avg_pool', 'fc_regression');

opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none', ...
    'Verbose', true, ...
    'InitialLearnRate', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'MaxEpochs', 7, ...
    'MiniBatchSize', 512, ...
    'Plots', 'training-progress', 'ExecutionEnvironment', 'parallel');

new_finalTrainInfo = {};

lgraph = layerGraph(model);
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:174) = freezeWeights(layers(1:174));
lgraph = createLgraphUsingConnections(layers,connections);


for i = 371:10:length(trainMatFiles)
    trainData = [];
    trainLabels = [];
    for j = i:min(i + 9, length(trainMatFiles))
        load(fullfile(trainMatFiles(j).folder, trainMatFiles(j).name), 'files', 'labels');
        trainData = cat(4, trainData, files);  
        trainLabels = [trainLabels; labels(1:size(files,4))];
        disp(j)
    end
    [model, traininfo] = trainNetwork(trainData, trainLabels, lgraph, opts);
    lgraph = layerGraph(model);
    
    new_finalTrainInfo{end + 1} = traininfo;
    
    save('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Workspaces/new_ResNet50_2', 'model', 'new_finalTrainInfo', '-v7.3');    
    
    clear trainLabels trainData traininfo;
end