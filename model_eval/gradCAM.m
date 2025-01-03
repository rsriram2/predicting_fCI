% Define a custom Grad-CAM function (simplified version)
function gradCAMMap = gradCAM(net, image, layerName, classIdx)
    % Forward pass through the network
    featureMaps = activations(net, image, layerName);
    
    % Backward pass to get the gradients
    dLdA = dlgradient(featureMaps(classIdx), featureMaps);
    
    % Compute the weights
    weights = mean(dLdA, [1 2]);
    
    % Compute the Grad-CAM map
    gradCAMMap = sum(weights .* featureMaps, 3);
    gradCAMMap = max(gradCAMMap, 0);
    gradCAMMap = imresize(gradCAMMap, [size(image, 1), size(image, 2)]);
end
