I = imread('entire.bmp'); % Replace with your image file
grayImage = rgb2gray(I); % Convert to grayscale
binaryImage = im2bw(grayImage,0.4);

% Define masks for removing noise
maskA = [0 0 0; 0 1 0; 0 0 0]; % For single white pixels
maskB = [0 0 0 0; 0 1 1 0; 0 1 1 0; 0 0 0 0]; % For 2x2 white squares

% Apply convolution to find matching pixels for each mask
resultA = conv2(binaryImage, maskA, 'same') == sum(maskA(:));
resultB = conv2(binaryImage, maskB,'same') == sum(maskB(:));

% Remove noise (update the binary image)
cleanedImage = binaryImage;
cleanedImage=cleanedImage.*resultA; % Remove single white pixels
cleanedImage=cleanedImage.*resultB; % Remove 2x2 white squares

thin_image=~bwmorph(cleanedImage,'thin',Inf);
figure;imshow(thin_image);title('Thinned Image');

% Display the results
figure;
subplot(1, 2, 1);
imshow(binaryImage);
title('Original Binary Image');

subplot(1, 2, 2);
imshow(cleanedImage);
title('Cleaned Image');
