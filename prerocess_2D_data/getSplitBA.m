structs = dir('*nii');
totalCount = 0;
patientIds = string(0);

for pID = 1:length(structs)
    patientIds(end+1) = structs(pID).name(1:end-4);
    totalCount = totalCount + 1;
end

patientIds = transpose(patientIds);
patientIds(1) = [];
shuffledIDs = patientIds(randperm(length(patientIds)));

numPatients = totalCount;

numTrain = round(0.70 * numPatients);
numTest = round(0.20 * numPatients);
numValidation = numPatients - numTrain - numTest;

trainIDs = shuffledIDs(1:numTrain);
testIDs = shuffledIDs(numTrain + 1:numTrain + numTest);
validationIDs = shuffledIDs(numTrain + numTest + 1:end);

trainingFiles = strings(0,2);
for i = 1:length(trainIDs)
    for j = 1:length(processedFiles)
            filePath = processedFiles(j);
            [~, name, ~] = fileparts(filePath);
            splitName = split(name, '.nii');
            patientId = splitName(1);
        if strcmpi(trainIDs(i), patientId)
            trainingFiles(end+1,1) = filePath;  
            for k = 1:height(pred)
                fullScanId = pred.ni_path(k);
                scanID = extractBefore(fullScanId, strlength(fullScanId)-3);
                if strcmpi(trainIDs(i),scanID)
                   trainingFiles(end, 2) = pred.age_years(k);
                end
            end
        end
    end
end

testFiles = strings(0,2);
for i = 1:length(testIDs)
    for j = 1:length(processedFiles)
            filePath = processedFiles(j);
            [~, name, ~] = fileparts(filePath);
            splitName = split(name, '.nii');
            patientId = splitName(1);
        if strcmpi(testIDs(i), patientId)
            testFiles(end+1,1) = filePath;  
            for k = 1:height(pred)
                fullScanId = pred.ni_path(k);
                scanID = extractBefore(fullScanId, strlength(fullScanId)-3);
                if strcmpi(testIDs(i),scanID)
                   testFiles(end, 2) = pred.age_years(k);
                end
            end
        end
    end
end

valFiles = strings(0,2);
for i = 1:length(validationIDs)
    for j = 1:length(processedFiles)
            filePath = processedFiles(j);
            [~, name, ~] = fileparts(filePath);
            splitName = split(name, '.nii');
            patientId = splitName(1);
        if strcmpi(validationIDs(i), patientId)
            valFiles(end+1,1) = filePath;  
            for k = 1:height(pred)
                fullScanId = pred.ni_path(k);
                scanID = extractBefore(fullScanId, strlength(fullScanId)-3);
                if strcmpi(validationIDs(i),scanID)
                   valFiles(end, 2) = pred.age_years(k);
                end
            end
        end
    end
end


modelTestFiles2= [];
modelTestlabels2 = [];

% numRemainingFiles = 19913 - 1;
% hWaitBar = waitbar(0, 'Processing images...');

for j = 1378:4118
    % waitbar((j - 1) / numRemainingFiles, hWaitBar);
    currentFilePath = shuffledData(j,1);
    currentFile = matRead(currentFilePath);
    targetSize = [227, 227];
    resizedFile = imresize(currentFile, targetSize);
    processImage = ind2rgb(im2uint8(mat2gray(resizedFile)), parula(256));
    modelTestFiles2 = cat(4, modelTestFiles2, processImage);
    modelTestlabels2(end+1) = shuffledData(j,2);
    disp(j);
end
% close(hWaitBar);
modelTestlabels2 = transpose(modelTestlabels2);


modelTestFiles2 = [];
modelTestlabels2 = [];

numRemainingFiles = 19913 - 13275;
hWaitBar = waitbar(0, 'Processing images...');

for j = 13275:19913
    waitbar((j - 13275) / numRemainingFiles, hWaitBar);
    currentFilePath = testFiles(j,1);
    currentFile = matRead(currentFilePath);
    targetSize = [227, 227];
    resizedFile = imresize(currentFile, targetSize);
    processImage = ind2rgb(im2uint8(mat2gray(resizedFile)), parula(256));
    modelTestFiles2 = cat(4, modelTestFiles2, processImage);
    modelTestlabels2(end+1) = testFiles(j,2);
end
close(hWaitBar);
modelTestlabels2 = transpose(modelTestlabels2);

modelTestFiles2 = [];
modelTestlabels2 = [];

numRemainingFiles = 3327 - 1;
hWaitBar = waitbar(0, 'Processing images...');

for j = 1:3327
    waitbar((j - 1) / numRemainingFiles, hWaitBar);
    currentFilePath = valFiles(j,1);
    currentFile = matRead(currentFilePath);
    targetSize = [227, 227];
    resizedFile = imresize(currentFile, targetSize);
    processImage = ind2rgb(im2uint8(mat2gray(resizedFile)), parula(256));
    modelTestFiles2 = cat(4, modelTestFiles2, processImage);
    modelTestlabels2(end+1) = valFiles(j,2);
end
close(hWaitBar);
modelTestlabels2 = transpose(modelTestlabels2);

concatenatedTestFiles = [];
concatenatedTestLabels = [];

testBatchOrder = [1, 2, 3, 4, 5, 6, 7];

for i = 1:7
    fileName = sprintf('shuffledTestBatch%d.mat', testBatchOrder(i));
    data = load(fileName);
    
    fileVarName = sprintf('modelTestFiles%d', testBatchOrder(i));
    labelVarName = sprintf('modelTestlabels%d', testBatchOrder(i));
    
    currentFiles = data.(fileVarName);
    currentLabels = data.(labelVarName);
    
    concatenatedTestFiles = cat(4, concatenatedTestFiles, currentFiles);
    concatenatedTestLabels = [concatenatedTestLabels; currentLabels];
    disp(i);
end

save('concatenatedTrainData.mat', 'concatenatedTestFiles', 'concatenatedTestLabels', '-v7.3');


trainData2 = cat(4, concatenatedTestFiles(35062:end), concatenatedTrain2Files);

numImages = 100;
resizedTrainFiles = zeros(384, 384, 3, numImages, 'like', concatenatedTrainFiles);

for i = 1:100
    resizedTrainFiles(:,:,:,i) = imresize(concatenatedTrainFiles(:,:,:,i), [384 384]);
    disp(i); % Display progress
end

numImages = size(shuffledTestFiles, 4);
resizedTrainFiles = zeros(224, 224, 3, numImages, 'like', shuffledTestFiles);
for i = 1:numImages
    resizedTrainFiles(:,:,:,i) = imresize(shuffledTestFiles(:,:,:,i), [224 224]);
    disp(i)
end
