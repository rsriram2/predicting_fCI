outputDir = '/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI/';
filteredDir = '/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/';

threshold = 0.44;

patientFolders = dir(fullfile(outputDir, 'IXI*'));
labelsTable = readtable('/mnt/argo/Workspaces/Staff/Helmet_Karim/Public/IXI/spreadsheets/ixi.xls');
labels = labelsTable{:, 'AGE'};
patientIDs = labelsTable{:, 'IXI_ID'};
numPatients = length(patientFolders);
indices = randperm(numPatients);

numTrain = round(0.70 * numPatients);
numTest = round(0.20 * numPatients);
numValidation = numPatients - numTrain - numTest;

trainIDs = indices(1:numTrain);
testIDs = indices(numTrain + 1:numTrain + numTest);
validationIDs = indices(numTrain + numTest + 1:end);

trainDir = '/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Train/';
testDir = '/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Test/';
valDir = '/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Validation/';

for k = 3:length(patientFolders)
    patientFolder = fullfile(outputDir, patientFolders(k).name);
    matFiles = dir(fullfile(patientFolder, '*.mat'));
    
    patientData = [];
    for j = 1:length(matFiles)
        matFile = fullfile(matFiles(j).folder, matFiles(j).name);
        data = load(matFile, 'files');
        files = data.files;
        validSlices = [];
        for i = 1:size(files, 4)
            if mean(mean(mean(files(:,:,:,i)))) > threshold
                validSlices = cat(4, validSlices, files(:,:,:,i));
            end
        end
        
        if ~isempty(validSlices)
            patientData = cat(4, patientData, validSlices);
       end
    end
        
    folderName = patientFolders(k).name;
    patientIDStr = folderName(4:6);
    patientID = str2double(patientIDStr);
    
    labelIdx = find(patientIDs == patientID);
    if ~isempty(labelIdx)
        Label = labels(labelIdx);
        patientLabel = repmat(Label, size(patientData,4),1);
        if ismember(k, trainIDs)
            saveDir = trainDir;
        elseif ismember(k, testIDs)
            saveDir = testDir;
        elseif ismember(k, validationIDs)
            saveDir = valDir;
        end
        matFileName = fullfile(saveDir, [folderName, '_filtered.mat']);
        m = matfile(matFileName, 'Writable', true);
        m.files = patientData;
        m.labels = patientLabel;
        disp(k)
    end
end
