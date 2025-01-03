options = trainingOptions('sgdm', ... 
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 1, ...  
    'MiniBatchSize', 512, ...
    'Shuffle', 'once', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','parallel',...
    'Verbose', false);

lgraph = layerGraph(model);

retrain2finalTrainInfo = {};

for i = 1:10:length(retrainMatFiles)
    trainData = [];
    trainLabels = [];
    for j = i:min(i + 9, length(retrainMatFiles))
        load(fullfile(retrainMatFiles(j).folder, retrainMatFiles(j).name), 'files', 'labels');
        trainData = cat(4, trainData, files);  
        trainLabels = [trainLabels; labels(1:size(files,4))];
        disp(j)
    end
    [model, traininfo] = trainNetwork(trainData, trainLabels, lgraph, options);
    lgraph = layerGraph(model);
    
    retrain2finalTrainInfo{end + 1} = traininfo;
    
    save('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Workspaces/retrained2ResNet50', 'model', 'retrain2finalTrainInfo', '-v7.3');    
    
    clear trainLabels trainData traininfo;
end