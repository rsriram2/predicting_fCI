input_scan_dir="/mnt/argo/Workspaces/Staff/Helmet_Karim/Public/IXI/mri"
input_transform_dir="/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Transformations"
output_dir="/mnt/argo/Workspaces/Students/Rushil_Srirambhatla/Public/BrainAge/Replicated_Data_2"

# Iterate over each scan file
for scan_file in "$input_scan_dir"/*.nii.gz; do
    scan_name=$(basename "$scan_file")
    echo "Processing scan: $scan_name"
    
    # Create a directory for the replicated data
    replicated_dir="$output_dir/${scan_name}"
    mkdir -p "$replicated_dir"

    # Get a random sample of 100 transformation files
    sampled_transform_files=$(find "$input_transform_dir" -type f -name "*.mat" | shuf | head -n 10)

    # Iterate over each line in sampled_transform_files
    while IFS= read -r transform_file; do
        # Process each file here
        transform_name=$(basename "$transform_file")
        echo "Applying transformation: $transform_name"
        
        # Apply transformation with flirt command
        output_file="$replicated_dir/${scan_name}_${transform_name%.mat}.nii"
        flirt -in "$scan_file" -ref "$scan_file" -applyxfm -init "$transform_file" -out "$output_file"
    done
done