load('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Workspaces/resnetver3_1.mat')
filterTrainResult{1,1} = 'PID';
filterTrainResult{1,2} = 'Actual_Age';
filterTrainResult{1,3} = 'Predicted_Age';
filterTrainResult{1,4} = 'Absolute_Error';

count = 2;
for i=2:length(trainResult)
    if trainResult{i,4} > 5
        filterTrainResult{count,1} = trainResult{i,1};
        filterTrainResult{count,2} = trainResult{i,2};
        filterTrainResult{count,3} = trainResult{i,3};
        filterTrainResult{count,4} = trainResult{i,4};
        disp(i)
        count = count + 1;
    end 
end

retrainMatFiles = [];
for j=2:length(filterTrainResult)
    currentID = filterTrainResult{j,1};
    correctID = [currentID, '-T1_filtered.mat'];
    index = find(arrayfun(@(x) strcmp(x.name, correctID), trainMatFiles));
    retrainMatFiles = [retrainMatFiles; trainMatFiles(index)];
end 