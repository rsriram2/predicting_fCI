function image_z = image_zscore(image)
image = double(image);

% use otsu threshold to come up with binary image
level = graythresh(image);
image_binary = imbinarize(image,level);

% mean and std deviation of image in threshold
image_avg = nanmean(image(image_binary));
image_std = nanstd(image(image_binary));

% calculate the z-score
image_z = (image - image_avg)/image_std;

end