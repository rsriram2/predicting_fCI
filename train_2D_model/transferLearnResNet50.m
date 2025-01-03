trainMatFiles = dir(fullfile('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Train/', '*.mat'));

options = trainingOptions('sgdm', ... 
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 5, ...  
    'MiniBatchSize', 512, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment','parallel',...
    'Verbose', false);

lgraph = layerGraph(model);

finalTrainInfo = {};

for i = 241:10:length(trainMatFiles)
    trainData = [];
    trainLabels = [];
    for j = i:min(i + 9, length(trainMatFiles))
        load(fullfile(trainMatFiles(j).folder, trainMatFiles(j).name), 'files', 'labels');
        trainData = cat(4, trainData, files);  
        trainLabels = [trainLabels; labels(1:size(files,4))];
        disp(j)
    end
    [model, traininfo] = trainNetwork(trainData, trainLabels, lgraph, options);
    lgraph = layerGraph(model);
    
    finalTrainInfo{end + 1} = traininfo;
    
    save('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Workspaces/transferLearnResNet50', 'model', 'finalTrainInfo', '-v7.3');    
    
    clear trainLabels trainData traininfo;
end


