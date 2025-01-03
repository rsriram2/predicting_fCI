patientFiles = dir("/mnt/argo/Workspaces/Staff/Helmet_Karim/Public/IXI/mri/*.nii.gz");
patientTestMatFolders = dir("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/IXI_filtered/Test/*.mat");

testResult2{1,1} = 'PID';
testResult2{1,2} = 'Actual_Age';
testResult2{1,3} = 'Predicted_Age';
testResult2{1,4} = 'Absolute_Error';

for i=1:length(patientTestMatFolders)
    testData = [];
    testLabels = [];
    currentPID = extractBefore(patientTestMatFolders(i).name, strlength(patientTestMatFolders(i).name)-12);
    fileIndex = find(strcmp({patientFiles.name}, strcat(currentPID, ".nii.gz")));
    file = niftiread(strcat(patientFiles(i).folder, "/", patientFiles(i).name));
    for j=1:10
        resized_file = imresize3(file, [224 224 224]);
        testData = cat(5,testData, resized_file);
        load(strcat(patientTestMatFolders(i).folder, "/",patientTestMatFolders(i).name), "labels");
        testLabels = vertcat(labels(1:1), testLabels);
    end
    testResult2{i+1,1} = currentPID;
    testResult2{i+1,2} = mean(testLabels(1:size(testData,5)));
    testResult2{i+1,3} = mean(predict(model, testData));
    testResult2{i+1,4} = abs(testResult2{i+1,2} - testResult2{i+1,3});
    clear testData testLabels
end


test_sum = 0;
for k=2:size(testResult2,1)
    test_sum = test_sum + testResult2{k,4};
end
test_mae2 = test_sum/(size(testResult2,1) - 1);

test_high_error_count2 = 0;
for h=2:size(testResult2,1)
    if(testResult2{h,4} > 5)
    test_high_error_count2 = test_high_error_count2 + 1;
    end
    clear h
end
test_percent_high_error2 = (test_high_error_count2/(size(testResult2,1)-1)) * 100;





