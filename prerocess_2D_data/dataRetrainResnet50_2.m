retrainMatFiles_2 = [];
for j=1:length(filterTrainResult)
    currentID = filterTrainResult{j,1};
    correctID = [currentID, '-T1_filtered.mat'];
    index = find(arrayfun(@(x) strcmp(x.name, correctID), trainMatFiles));
    retrainMatFiles_2 = [retrainMatFiles_2; trainMatFiles(index)];
end 