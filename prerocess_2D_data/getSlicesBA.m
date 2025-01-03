fileList = dir('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/ReplicatedNIFTI/*nii');
parentFolder = '/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/ReplicatedNIFTI/';
for i = 1:numel(fileList)
    currentFolder = fileList(i).name;
    niftiFiles = dir(fullfile(parentFolder,currentFolder, '*nii'));

    for j = 1:numel(niftiFiles)
        pos = strfind(niftiFiles(j).name, '_');
        outputFolder = fullfile('/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Slices2', fileList(i).name,niftiFiles(j).name(pos(2) + 1:end-4));
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
        end
        currentFile = fullfile(parentFolder, currentFolder, niftiFiles(j).name);
        C = niftiread(currentFile);
        for k = 1:size(C, 3)
            im_new = C(:, :, k);
            name = strcat(niftiFiles(j).name(1:end-4), '_',num2str(k),'.mat');
            fullFileName = fullfile(outputFolder, name);
            save(fullFileName, 'im_new');
        end
    end
end
