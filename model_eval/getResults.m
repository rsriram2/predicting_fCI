scatter(averagedResults2(:,2), results3(:,3));
residuals = results3(:,2) - results3(:,1);
scatter(results3(:,1), residuals);

% Extract unique patient IDs
uniquePatientIDs = unique(results3(:, 3));

% Initialize a new cell array for the results
numUniquePatients = length(uniquePatientIDs);
averagedResults = cell(numUniquePatients, 3);

% Loop through each unique patient ID
for i = 1:numUniquePatients
    patientID = uniquePatientIDs{i};
    
    % Find the rows corresponding to the current patient ID
    isCurrentPatient = strcmp(results3(:, 3), patientID);
    
    % Extract actual and predicted ages for the current patient ID
    actualAges = results3(isCurrentPatient, 2);
    predictedAges = results3(isCurrentPatient, 1);
    
    % Convert cells to numeric arrays
    actualAges = cellfun(@(x) str2double(x), actualAges);
    predictedAges = cellfun(@(x) str2double(x), predictedAges);
    
    % Calculate average actual age and average predicted age
    avgActualAge = mean(actualAges);
    avgPredictedAge = mean(predictedAges);
    
    % Store the results
    averagedResults{i, 1} = patientID;
    averagedResults{i, 2} = avgActualAge;
    averagedResults{i, 3} = avgPredictedAge;
end
