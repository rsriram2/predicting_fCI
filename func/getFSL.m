directory = '/Volumes/SRI/AD Prediction Project/Slices/Train';
fileList = dir(fullfile(directory, '*.mat'));
[~, fileName, fileExt] = fileparts(fileList(i).name);

for k = 1:500
    T = -3 + (3+3)*rand(1,3);
    R = -0.25 + (0.25+0.25)*rand(1,3);
    S = [1 1 1];
    M = makeFslXfmMatrix(T, R, S, ['/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Transformations', num2str(k),'.mat']);
end

