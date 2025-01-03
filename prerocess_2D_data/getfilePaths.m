patientList = dir;
filePaths = string(0);
for i = 3:length(patientList)
    transformation = dir(patientList(i).name);
    for j = 3:length(transformation)
        slices = dir([transformation(j).folder, '/', transformation(j).name]);
        for k = 3:length(slices)
        fileName = string([slices(k).folder, '/', slices(k).name]);
        filePaths(end+1) = fileName;
        end
    end
end

filePaths = transpose(filePaths);
filePaths(1) = [];

processedFiles = string(0);

for file = 1:length(filePaths)
    image = matRead(filePaths(file));
    if mean(image(:)) ~= 0
        processedFiles(end+1) = filePaths(file);
    end
end

processedFiles = transpose(processedFiles);
processedFiles(1) = [];