patientFolders = dir("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/ReplicatedNIFTI2/*.nii.gz");
patientMatFolders = dir("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Train/*.mat");
patientMatFolders(133) = [];
patientMatFolders(197) = [];
patientMatFolders(202) = [];
patientMatFolders(368) = [];
for i=1:length(patientMatFolders)
    trainData = [];
    trainLabels = [];
    currentPID = extractBefore(patientMatFolders(i).name, strlength(patientMatFolders(i).name)-12);
    fileIndex = find(strcmp({patientFolders.name}, strcat(currentPID, ".nii.gz")));
    files = dir(strcat(patientFolders(fileIndex).folder, "/", patientFolders(fileIndex).name, "/*nii.gz"));
    for j=1:length(files)
        file = niftiread(strcat(files(j).folder, "/", files(j).name));
        file = permute(file, [3, 1, 2]);
        resized_file = imresize3(file, [224 224 224]);
        trainData = cat(5,trainData, resized_file);
        load(strcat(patientMatFolders(i).folder, "/",patientMatFolders(i).name), "labels");
        trainLabels = vertcat(labels(1:1), trainLabels);
    end
    disp(i)
    save(strcat("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/patientVolumes/", currentPID, ".mat"),"trainData", "trainLabels")
end

lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

numOutputs = 1; 
newLayers = [
    fullyConnectedLayer(numOutputs, 'Name', 'fc_regression')
    regressionLayer('Name', 'regressionoutput')];

lgraph = addLayers(lgraph, newLayers);

lgraph = connectLayers(lgraph, 'avg_pool', 'fc_regression');
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:174) = freezeWeights(layers(1:174));
lgraph = createLgraphUsingConnections(layers,connections);

opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none', ...
    'Verbose', true, ...
    'InitialLearnRate', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'MaxEpochs', 7, ...
    'MiniBatchSize', 1, ...
    'Plots', 'training-progress', 'ExecutionEnvironment', 'parallel');

new_finalTrainInfo = {};
    
trainMatFiles = dir(fullfile('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/patientVolumes/', '*.mat'));


for i = 1:length(trainMatFiles)
    load(fullfile(trainMatFiles(i).folder, trainMatFiles(i).name), 'trainData', 'trainLabels');
    disp(i)
    for j =1:size(trainData, 4)
        [model, traininfo] = trainNetwork(trainData, trainLabels, lgraph, opts);
        lgraph = layerGraph(model);
        new_finalTrainInfo{end + 1} = traininfo;
        save('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Workspaces/new_ResNet50_3D', 'model', 'new_finalTrainInfo', '-v7.3');
        clear trainData trainLabels traininfo;
    end
end

