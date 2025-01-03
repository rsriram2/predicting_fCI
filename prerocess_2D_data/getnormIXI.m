patientFiles = dir("/mnt/argo/Workspaces/Staff/Helmet_Karim/Public/IXI/mri/*.nii.gz");
patientFiles(193) = [];
patientFiles(290) = [];
patientFiles(301) = [];
patientFiles(386) = [];
patientFiles(551) = [];

for i=1:length(patientFiles)
    image_filepath = strcat(patientFiles(i).folder, '/', patientFiles(i).name);
    hdr = load_untouch_nii(image_filepath);
    image_z = image_zscore(hdr.img);
    hdr.img = image_z;
    hdr.hdr.dime.bitpix = 64;
    hdr.hdr.dime.datatype = 64;
    hdr.hdr.dime.glmax = max(max(max(image_z)));
    hdr.hdr.dime.glmin = min(min(min(image_z)));
    save_untouch_nii(hdr,strcat("/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/normIXI/", patientFiles(i).name(1:end-3)));
end


