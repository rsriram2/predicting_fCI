numRows = size(testFiles, 1);

randomOrder = randperm(numRows);

shuffledData = testFiles(randomOrder, :);