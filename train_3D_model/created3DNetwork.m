params = load("params.mat");
lgraph = layerGraph();
tempLayers = [
    image3dInputLayer([224 224 224 1],"Name","input_1","Mean",params.input_1.Mean)
    convolution3dLayer([7 7 7],64,"Name","conv1","Padding",[3 3 3;3 3 3],"Stride",[2 2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001,"Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
    reluLayer("Name","activation_1_relu")
    maxPooling3dLayer([3 3 3],"Name","max_pooling3d_1","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res2a_branch1","Bias",params.res2a_branch1.Bias,"Weights",params.res2a_branch1.Weights)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001,"Offset",params.bn2a_branch1.Offset,"Scale",params.bn2a_branch1.Scale,"TrainedMean",params.bn2a_branch1.TrainedMean,"TrainedVariance",params.bn2a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],64,"Name","res2a_branch2a","Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001,"Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","activation_2_relu")
    convolution3dLayer([3 3 3],64,"Name","res2a_branch2b","Padding","same","Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001,"Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)
    reluLayer("Name","activation_3_relu")
    convolution3dLayer([1 1 1],256,"Name","res2a_branch2c","Bias",params.res2a_branch2c.Bias,"Weights",params.res2a_branch2c.Weights)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001,"Offset",params.bn2a_branch2c.Offset,"Scale",params.bn2a_branch2c.Scale,"TrainedMean",params.bn2a_branch2c.TrainedMean,"TrainedVariance",params.bn2a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],64,"Name","res2b_branch2a","Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001,"Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","activation_5_relu")
    convolution3dLayer([3 3 3],64,"Name","res2b_branch2b","Padding","same","Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001,"Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)
    reluLayer("Name","activation_6_relu")
    convolution3dLayer([1 1 1],256,"Name","res2b_branch2c","Bias",params.res2b_branch2c.Bias,"Weights",params.res2b_branch2c.Weights)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001,"Offset",params.bn2b_branch2c.Offset,"Scale",params.bn2b_branch2c.Scale,"TrainedMean",params.bn2b_branch2c.TrainedMean,"TrainedVariance",params.bn2b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],64,"Name","res2c_branch2a","Bias",params.res2c_branch2a.Bias,"Weights",params.res2c_branch2a.Weights)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001,"Offset",params.bn2c_branch2a.Offset,"Scale",params.bn2c_branch2a.Scale,"TrainedMean",params.bn2c_branch2a.TrainedMean,"TrainedVariance",params.bn2c_branch2a.TrainedVariance)
    reluLayer("Name","activation_8_relu")
    convolution3dLayer([3 3 3],64,"Name","res2c_branch2b","Padding","same","Bias",params.res2c_branch2b.Bias,"Weights",params.res2c_branch2b.Weights)
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001,"Offset",params.bn2c_branch2b.Offset,"Scale",params.bn2c_branch2b.Scale,"TrainedMean",params.bn2c_branch2b.TrainedMean,"TrainedVariance",params.bn2c_branch2b.TrainedVariance)
    reluLayer("Name","activation_9_relu")
    convolution3dLayer([1 1 1],256,"Name","res2c_branch2c","Bias",params.res2c_branch2c.Bias,"Weights",params.res2c_branch2c.Weights)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001,"Offset",params.bn2c_branch2c.Offset,"Scale",params.bn2c_branch2c.Scale,"TrainedMean",params.bn2c_branch2c.TrainedMean,"TrainedVariance",params.bn2c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","res3a_branch2a","Stride",[2 2 2],"Bias",params.res3a_branch2a.Bias,"Weights",params.res3a_branch2a.Weights)
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001,"Offset",params.bn3a_branch2a.Offset,"Scale",params.bn3a_branch2a.Scale,"TrainedMean",params.bn3a_branch2a.TrainedMean,"TrainedVariance",params.bn3a_branch2a.TrainedVariance)
    reluLayer("Name","activation_11_relu")
    convolution3dLayer([3 3 3],128,"Name","res3a_branch2b","Padding","same","Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001,"Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)
    reluLayer("Name","activation_12_relu")
    convolution3dLayer([1 1 1],512,"Name","res3a_branch2c","Bias",params.res3a_branch2c.Bias,"Weights",params.res3a_branch2c.Weights)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001,"Offset",params.bn3a_branch2c.Offset,"Scale",params.bn3a_branch2c.Scale,"TrainedMean",params.bn3a_branch2c.TrainedMean,"TrainedVariance",params.bn3a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],512,"Name","res3a_branch1","Stride",[2 2 2],"Bias",params.res3a_branch1.Bias,"Weights",params.res3a_branch1.Weights)
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001,"Offset",params.bn3a_branch1.Offset,"Scale",params.bn3a_branch1.Scale,"TrainedMean",params.bn3a_branch1.TrainedMean,"TrainedVariance",params.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","res3b_branch2a","Bias",params.res3b_branch2a.Bias,"Weights",params.res3b_branch2a.Weights)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001,"Offset",params.bn3b_branch2a.Offset,"Scale",params.bn3b_branch2a.Scale,"TrainedMean",params.bn3b_branch2a.TrainedMean,"TrainedVariance",params.bn3b_branch2a.TrainedVariance)
    reluLayer("Name","activation_14_relu")
    convolution3dLayer([3 3 3],128,"Name","res3b_branch2b","Padding","same","Bias",params.res3b_branch2b.Bias,"Weights",params.res3b_branch2b.Weights)
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001,"Offset",params.bn3b_branch2b.Offset,"Scale",params.bn3b_branch2b.Scale,"TrainedMean",params.bn3b_branch2b.TrainedMean,"TrainedVariance",params.bn3b_branch2b.TrainedVariance)
    reluLayer("Name","activation_15_relu")
    convolution3dLayer([1 1 1],512,"Name","res3b_branch2c","Bias",params.res3b_branch2c.Bias,"Weights",params.res3b_branch2c.Weights)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001,"Offset",params.bn3b_branch2c.Offset,"Scale",params.bn3b_branch2c.Scale,"TrainedMean",params.bn3b_branch2c.TrainedMean,"TrainedVariance",params.bn3b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","res3c_branch2a","Bias",params.res3c_branch2a.Bias,"Weights",params.res3c_branch2a.Weights)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001,"Offset",params.bn3c_branch2a.Offset,"Scale",params.bn3c_branch2a.Scale,"TrainedMean",params.bn3c_branch2a.TrainedMean,"TrainedVariance",params.bn3c_branch2a.TrainedVariance)
    reluLayer("Name","activation_17_relu")
    convolution3dLayer([3 3 3],128,"Name","res3c_branch2b","Padding","same","Bias",params.res3c_branch2b.Bias,"Weights",params.res3c_branch2b.Weights)
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001,"Offset",params.bn3c_branch2b.Offset,"Scale",params.bn3c_branch2b.Scale,"TrainedMean",params.bn3c_branch2b.TrainedMean,"TrainedVariance",params.bn3c_branch2b.TrainedVariance)
    reluLayer("Name","activation_18_relu")
    convolution3dLayer([1 1 1],512,"Name","res3c_branch2c","Bias",params.res3c_branch2c.Bias,"Weights",params.res3c_branch2c.Weights)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001,"Offset",params.bn3c_branch2c.Offset,"Scale",params.bn3c_branch2c.Scale,"TrainedMean",params.bn3c_branch2c.TrainedMean,"TrainedVariance",params.bn3c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","res3d_branch2a","Bias",params.res3d_branch2a.Bias,"Weights",params.res3d_branch2a.Weights)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001,"Offset",params.bn3d_branch2a.Offset,"Scale",params.bn3d_branch2a.Scale,"TrainedMean",params.bn3d_branch2a.TrainedMean,"TrainedVariance",params.bn3d_branch2a.TrainedVariance)
    reluLayer("Name","activation_20_relu")
    convolution3dLayer([3 3 3],128,"Name","res3d_branch2b","Padding","same","Bias",params.res3d_branch2b.Bias,"Weights",params.res3d_branch2b.Weights)
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001,"Offset",params.bn3d_branch2b.Offset,"Scale",params.bn3d_branch2b.Scale,"TrainedMean",params.bn3d_branch2b.TrainedMean,"TrainedVariance",params.bn3d_branch2b.TrainedVariance)
    reluLayer("Name","activation_21_relu")
    convolution3dLayer([1 1 1],512,"Name","res3d_branch2c","Bias",params.res3d_branch2c.Bias,"Weights",params.res3d_branch2c.Weights)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001,"Offset",params.bn3d_branch2c.Offset,"Scale",params.bn3d_branch2c.Scale,"TrainedMean",params.bn3d_branch2c.TrainedMean,"TrainedVariance",params.bn3d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4a_branch2a","Stride",[2 2 2],"Bias",params.res4a_branch2a.Bias,"Weights",params.res4a_branch2a.Weights)
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001,"Offset",params.bn4a_branch2a.Offset,"Scale",params.bn4a_branch2a.Scale,"TrainedMean",params.bn4a_branch2a.TrainedMean,"TrainedVariance",params.bn4a_branch2a.TrainedVariance)
    reluLayer("Name","activation_23_relu")
    convolution3dLayer([3 3 3],256,"Name","res4a_branch2b","Padding","same","Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001,"Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)
    reluLayer("Name","activation_24_relu")
    convolution3dLayer([1 1 1],1024,"Name","res4a_branch2c","Bias",params.res4a_branch2c.Bias,"Weights",params.res4a_branch2c.Weights)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001,"Offset",params.bn4a_branch2c.Offset,"Scale",params.bn4a_branch2c.Scale,"TrainedMean",params.bn4a_branch2c.TrainedMean,"TrainedVariance",params.bn4a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],1024,"Name","res4a_branch1","Stride",[2 2 2],"Bias",params.res4a_branch1.Bias,"Weights",params.res4a_branch1.Weights)
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001,"Offset",params.bn4a_branch1.Offset,"Scale",params.bn4a_branch1.Scale,"TrainedMean",params.bn4a_branch1.TrainedMean,"TrainedVariance",params.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4b_branch2a","Bias",params.res4b_branch2a.Bias,"Weights",params.res4b_branch2a.Weights)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001,"Offset",params.bn4b_branch2a.Offset,"Scale",params.bn4b_branch2a.Scale,"TrainedMean",params.bn4b_branch2a.TrainedMean,"TrainedVariance",params.bn4b_branch2a.TrainedVariance)
    reluLayer("Name","activation_26_relu")
    convolution3dLayer([3 3 3],256,"Name","res4b_branch2b","Padding","same","Bias",params.res4b_branch2b.Bias,"Weights",params.res4b_branch2b.Weights)
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001,"Offset",params.bn4b_branch2b.Offset,"Scale",params.bn4b_branch2b.Scale,"TrainedMean",params.bn4b_branch2b.TrainedMean,"TrainedVariance",params.bn4b_branch2b.TrainedVariance)
    reluLayer("Name","activation_27_relu")
    convolution3dLayer([1 1 1],1024,"Name","res4b_branch2c","Bias",params.res4b_branch2c.Bias,"Weights",params.res4b_branch2c.Weights)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001,"Offset",params.bn4b_branch2c.Offset,"Scale",params.bn4b_branch2c.Scale,"TrainedMean",params.bn4b_branch2c.TrainedMean,"TrainedVariance",params.bn4b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4c_branch2a","Bias",params.res4c_branch2a.Bias,"Weights",params.res4c_branch2a.Weights)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001,"Offset",params.bn4c_branch2a.Offset,"Scale",params.bn4c_branch2a.Scale,"TrainedMean",params.bn4c_branch2a.TrainedMean,"TrainedVariance",params.bn4c_branch2a.TrainedVariance)
    reluLayer("Name","activation_29_relu")
    convolution3dLayer([3 3 3],256,"Name","res4c_branch2b","Padding","same","Bias",params.res4c_branch2b.Bias,"Weights",params.res4c_branch2b.Weights)
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001,"Offset",params.bn4c_branch2b.Offset,"Scale",params.bn4c_branch2b.Scale,"TrainedMean",params.bn4c_branch2b.TrainedMean,"TrainedVariance",params.bn4c_branch2b.TrainedVariance)
    reluLayer("Name","activation_30_relu")
    convolution3dLayer([1 1 1],1024,"Name","res4c_branch2c","Bias",params.res4c_branch2c.Bias,"Weights",params.res4c_branch2c.Weights)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001,"Offset",params.bn4c_branch2c.Offset,"Scale",params.bn4c_branch2c.Scale,"TrainedMean",params.bn4c_branch2c.TrainedMean,"TrainedVariance",params.bn4c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4d_branch2a","Bias",params.res4d_branch2a.Bias,"Weights",params.res4d_branch2a.Weights)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001,"Offset",params.bn4d_branch2a.Offset,"Scale",params.bn4d_branch2a.Scale,"TrainedMean",params.bn4d_branch2a.TrainedMean,"TrainedVariance",params.bn4d_branch2a.TrainedVariance)
    reluLayer("Name","activation_32_relu")
    convolution3dLayer([3 3 3],256,"Name","res4d_branch2b","Padding","same","Bias",params.res4d_branch2b.Bias,"Weights",params.res4d_branch2b.Weights)
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001,"Offset",params.bn4d_branch2b.Offset,"Scale",params.bn4d_branch2b.Scale,"TrainedMean",params.bn4d_branch2b.TrainedMean,"TrainedVariance",params.bn4d_branch2b.TrainedVariance)
    reluLayer("Name","activation_33_relu")
    convolution3dLayer([1 1 1],1024,"Name","res4d_branch2c","Bias",params.res4d_branch2c.Bias,"Weights",params.res4d_branch2c.Weights)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001,"Offset",params.bn4d_branch2c.Offset,"Scale",params.bn4d_branch2c.Scale,"TrainedMean",params.bn4d_branch2c.TrainedMean,"TrainedVariance",params.bn4d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4e_branch2a","Bias",params.res4e_branch2a.Bias,"Weights",params.res4e_branch2a.Weights)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001,"Offset",params.bn4e_branch2a.Offset,"Scale",params.bn4e_branch2a.Scale,"TrainedMean",params.bn4e_branch2a.TrainedMean,"TrainedVariance",params.bn4e_branch2a.TrainedVariance)
    reluLayer("Name","activation_35_relu")
    convolution3dLayer([3 3 3],256,"Name","res4e_branch2b","Padding","same","Bias",params.res4e_branch2b.Bias,"Weights",params.res4e_branch2b.Weights)
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001,"Offset",params.bn4e_branch2b.Offset,"Scale",params.bn4e_branch2b.Scale,"TrainedMean",params.bn4e_branch2b.TrainedMean,"TrainedVariance",params.bn4e_branch2b.TrainedVariance)
    reluLayer("Name","activation_36_relu")
    convolution3dLayer([1 1 1],1024,"Name","res4e_branch2c","Bias",params.res4e_branch2c.Bias,"Weights",params.res4e_branch2c.Weights)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001,"Offset",params.bn4e_branch2c.Offset,"Scale",params.bn4e_branch2c.Scale,"TrainedMean",params.bn4e_branch2c.TrainedMean,"TrainedVariance",params.bn4e_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","res4f_branch2a","Bias",params.res4f_branch2a.Bias,"Weights",params.res4f_branch2a.Weights)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001,"Offset",params.bn4f_branch2a.Offset,"Scale",params.bn4f_branch2a.Scale,"TrainedMean",params.bn4f_branch2a.TrainedMean,"TrainedVariance",params.bn4f_branch2a.TrainedVariance)
    reluLayer("Name","activation_38_relu")
    convolution3dLayer([3 3 3],256,"Name","res4f_branch2b","Padding","same","Bias",params.res4f_branch2b.Bias,"Weights",params.res4f_branch2b.Weights)
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001,"Offset",params.bn4f_branch2b.Offset,"Scale",params.bn4f_branch2b.Scale,"TrainedMean",params.bn4f_branch2b.TrainedMean,"TrainedVariance",params.bn4f_branch2b.TrainedVariance)
    reluLayer("Name","activation_39_relu")
    convolution3dLayer([1 1 1],1024,"Name","res4f_branch2c","Bias",params.res4f_branch2c.Bias,"Weights",params.res4f_branch2c.Weights)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001,"Offset",params.bn4f_branch2c.Offset,"Scale",params.bn4f_branch2c.Scale,"TrainedMean",params.bn4f_branch2c.TrainedMean,"TrainedVariance",params.bn4f_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],2048,"Name","res5a_branch1","Stride",[2 2 2],"Bias",params.res5a_branch1.Bias,"Weights",params.res5a_branch1.Weights)
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001,"Offset",params.bn5a_branch1.Offset,"Scale",params.bn5a_branch1.Scale,"TrainedMean",params.bn5a_branch1.TrainedMean,"TrainedVariance",params.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],512,"Name","res5a_branch2a","Stride",[2 2 2],"Bias",params.res5a_branch2a.Bias,"Weights",params.res5a_branch2a.Weights)
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001,"Offset",params.bn5a_branch2a.Offset,"Scale",params.bn5a_branch2a.Scale,"TrainedMean",params.bn5a_branch2a.TrainedMean,"TrainedVariance",params.bn5a_branch2a.TrainedVariance)
    reluLayer("Name","activation_41_relu")
    convolution3dLayer([3 3 3],512,"Name","res5a_branch2b","Padding","same","Bias",params.res5a_branch2b.Bias,"Weights",params.res5a_branch2b.Weights)
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001,"Offset",params.bn5a_branch2b.Offset,"Scale",params.bn5a_branch2b.Scale,"TrainedMean",params.bn5a_branch2b.TrainedMean,"TrainedVariance",params.bn5a_branch2b.TrainedVariance)
    reluLayer("Name","activation_42_relu")
    convolution3dLayer([1 1 1],2048,"Name","res5a_branch2c","Bias",params.res5a_branch2c.Bias,"Weights",params.res5a_branch2c.Weights)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001,"Offset",params.bn5a_branch2c.Offset,"Scale",params.bn5a_branch2c.Scale,"TrainedMean",params.bn5a_branch2c.TrainedMean,"TrainedVariance",params.bn5a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],512,"Name","res5b_branch2a","Bias",params.res5b_branch2a.Bias,"Weights",params.res5b_branch2a.Weights)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001,"Offset",params.bn5b_branch2a.Offset,"Scale",params.bn5b_branch2a.Scale,"TrainedMean",params.bn5b_branch2a.TrainedMean,"TrainedVariance",params.bn5b_branch2a.TrainedVariance)
    reluLayer("Name","activation_44_relu")
    convolution3dLayer([3 3 3],512,"Name","res5b_branch2b","Padding","same","Bias",params.res5b_branch2b.Bias,"Weights",params.res5b_branch2b.Weights)
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001,"Offset",params.bn5b_branch2b.Offset,"Scale",params.bn5b_branch2b.Scale,"TrainedMean",params.bn5b_branch2b.TrainedMean,"TrainedVariance",params.bn5b_branch2b.TrainedVariance)
    reluLayer("Name","activation_45_relu")
    convolution3dLayer([1 1 1],2048,"Name","res5b_branch2c","Bias",params.res5b_branch2c.Bias,"Weights",params.res5b_branch2c.Weights)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001,"Offset",params.bn5b_branch2c.Offset,"Scale",params.bn5b_branch2c.Scale,"TrainedMean",params.bn5b_branch2c.TrainedMean,"TrainedVariance",params.bn5b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    convolution3dLayer([1 1 1],512,"Name","res5c_branch2a","Bias",params.res5c_branch2a.Bias,"Weights",params.res5c_branch2a.Weights)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001,"Offset",params.bn5c_branch2a.Offset,"Scale",params.bn5c_branch2a.Scale,"TrainedMean",params.bn5c_branch2a.TrainedMean,"TrainedVariance",params.bn5c_branch2a.TrainedVariance)
    reluLayer("Name","activation_47_relu")
    convolution3dLayer([3 3 3],512,"Name","res5c_branch2b","Padding","same","Bias",params.res5c_branch2b.Bias,"Weights",params.res5c_branch2b.Weights)
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001,"Offset",params.bn5c_branch2b.Offset,"Scale",params.bn5c_branch2b.Scale,"TrainedMean",params.bn5c_branch2b.TrainedMean,"TrainedVariance",params.bn5c_branch2b.TrainedVariance)
    reluLayer("Name","activation_48_relu")
    convolution3dLayer([1 1 1],2048,"Name","res5c_branch2c","Bias",params.res5c_branch2c.Bias,"Weights",params.res5c_branch2c.Weights)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001,"Offset",params.bn5c_branch2c.Offset,"Scale",params.bn5c_branch2c.Scale,"TrainedMean",params.bn5c_branch2c.TrainedMean,"TrainedVariance",params.bn5c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);
tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")
    globalAveragePooling3dLayer("Name","avg_pool")
    fullyConnectedLayer(1000,"Name","fc1000","Bias",params.fc1000.Bias,"Weights",params.fc1000.Weights)
    softmaxLayer("Name","fc1000_softmax")
    classificationLayer("Name","ClassificationLayer_fc1000","Classes",params.ClassificationLayer_fc1000.Classes)];
lgraph = addLayers(lgraph,tempLayers);
% clean up helper variable
clear tempLayers;
lgraph = connectLayers(lgraph,"max_pooling3d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"max_pooling3d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

numOutputs = 1; 
newLayers = [
    batchNormalizationLayer('Name', 'batchnorm')
    reluLayer('Name', 'relu')
    fullyConnectedLayer(numOutputs, 'Name', 'fc_regression')
    regressionLayer('Name', 'regressionoutput')];

lgraph = addLayers(lgraph, newLayers);

lgraph = connectLayers(lgraph, 'avg_pool', 'batchnorm');

layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:174) = freezeWeights(layers(1:174));
lgraph = createLgraphUsingConnections(layers,connections);

opts = trainingOptions('adam', ...
    'LearnRateSchedule', 'none', ...
    'Verbose', true, ...
    'InitialLearnRate', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'MaxEpochs', 7, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','parallel', ...
    'MiniBatchSize', 30);

% try sgdm
% increase minibatchsize to 30
% normalize intensity ranges ("white stripes" method or standardize actual intensity o2 mean and sd of voxels>0 subtract by mean divide by standard mask_otsu)
% [img-mean(img(mask_otsu))] / std(img(mask_otsu))
% input is image and output is standarized image
% classification 