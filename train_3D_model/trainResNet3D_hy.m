new_finalTrainInfo = {};
patientFiles = dir("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/normIXI/*.nii");
patientMatFolders = dir("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Train/*.mat");
patientMatFolders(133) = [];
patientMatFolders(197) = [];
patientMatFolders(202) = [];
patientMatFolders(368) = [];

for i=1:100:length(patientMatFolders)
    trainData = [];
    trainLabels = [];
    for j=i:min(i+99, length(patientMatFolders))
    currentPID = extractBefore(patientMatFolders(j).name, strlength(patientMatFolders(j).name)-12);
    fileIndex = find(strcmp({patientFiles.name}, strcat(currentPID, ".nii")));
    hdr = load_untouch_nii(strcat(patientFiles(j).folder, "/", patientFiles(j).name));
    file = hdr.img;
        for h=1:10
            tform = randomAffine3d('Scale',[0.85,1.15],...
            'Rotation',[-15 +15],...
            'XTranslation',[-15 15],...
            'YTranslation',[-15 15],...
            'ZTranslation',[-15 15]);
        temp = imwarp(file,tform);
        resized_file = imresize3(temp, [224 224 224]);
        trainData = cat(5,trainData, resized_file);
        load(strcat(patientMatFolders(j).folder, "/",patientMatFolders(j).name), "labels");
        trainLabels = vertcat(labels(1:1), trainLabels);
        end
        disp(j)
    end
    disp(i)
    [model, traininfo] = trainNetwork(trainData, trainLabels, lgraph, opts);
    lgraph = layerGraph(model);
    new_finalTrainInfo{end + 1} = traininfo;    
    save('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Workspaces/new_ResNet50_3D_hy', 'model', 'new_finalTrainInfo', '-v7.3');
    clear trainData trainLabels traininfo;
end