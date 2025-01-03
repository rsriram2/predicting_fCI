testMatFiles = dir(fullfile('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Test/', '*.mat'));
retrainedTestResult2{1,1} = 'PID';
retrainedTestResult2{1,2} = 'Actual_Age';
retrainedTestResult2{1,3} = 'Predicted_Age';
retrainedTestResult2{1,4} = 'Absolute_Error';

for i=1:length(testMatFiles)
    load(testMatFiles(i).name)
    retrainedTestResult2{i+1,1} = testMatFiles(i).name(1:end-16);
    retrainedTestResult2{i+1,2} = mean(labels(1:size(files,4)));
    retrainedTestResult2{i+1,3} = mean(predict(model, files));
end


for j=2:size(retrainedTestResult2,1)
    retrainedTestResult2{j,4} = abs(retrainedTestResult2{j,2} - retrainedTestResult2{j,3});
end

retrainedTestsum2 = 0;
for k=2:size(retrainedTestResult2,1)
    retrainedTestsum2 = retrainedTestsum2 + retrainedTestResult2{k,4};
    clear k
end
retrainedTest_mae2 = retrainedTestsum2/(size(retrainedTestResult2,1) - 1);

retrainedTest_high_error_count2 = 0;
for h=2:size(retrainedTestResult2,1)
    if(retrainedTestResult2{h,4} > 5)
    retrainedTest_high_error_count2 = retrainedTest_high_error_count2 + 1;
    end
    clear h
end
retrainedTest_percent_high_error2 = (retrainedTest_high_error_count2/(size(retrainedTestResult2,1)-1)) * 100;


trainMatFiles = dir(fullfile('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Train/', '*.mat'));
retrainedTrainResult2{1,1} = 'PID';
retrainedTrainResult2{1,2} = 'Actual_Age';
retrainedTrainResult2{1,3} = 'Predicted_Age';
retrainedTrainResult2{1,4} = 'Absolute_Error';

for i=1:length(trainMatFiles)
    load(trainMatFiles(i).name)
    retrainedTrainResult2{i+1,1} = trainMatFiles(i).name(1:end-16);
    retrainedTrainResult2{i+1,2} = mean(labels(1:size(files,4)));
    retrainedTrainResult2{i+1,3} = mean(predict(model, files));
    retrainedTrainResult2{i+1,4} = abs(retrainedTrainResult2{i+1,2} - retrainedTrainResult2{i+1,3});
end

retrainedTrain_sum2 = 0;
for k=2:size(retrainedTrainResult2,1)
    retrainedTrain_sum2 = retrainedTrain_sum2 + retrainedTrainResult2{k,4};
end
retreainTrain_mae2 = retrainedTrain_sum2/(size(retrainedTrainResult2,1) - 1);

retrainedTrain_high_error_count2 = 0;
for h=2:size(retrainedTrainResult2,1)
    if(retrainedTrainResult2{h,4} > 5)
    retrainedTrain_high_error_count2 = retrainedTrain_high_error_count2 + 1;
    end
    clear h
end
retrainedTrain_percent_high_error2 = (retrainedTrain_high_error_count2/(size(retrainedTrainResult2,1)-1)) * 100;