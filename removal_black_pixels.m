% Load the binary fingerprint image
I = imread('entire.bmp'); % Replace with your image file
grayImage = rgb2gray(I); % Convert to grayscale


% Step 1: Remove black noise with square frames (sizes 4x4 to 9x9)
cleanedImage = binaryImage;
for frameSize = 4:9
    mask = ones(frameSize); % Create square mask
    convolved = conv2(binaryImage, mask, 'same'); % Convolve with image
    threshold = frameSize^2 - 1; % Define threshold
    cleanedImage(convolved >= threshold) = 1; % Remove black spots
end

% Step 2: Remove small black noise using 3x3 frame
mask3x3 = [1 1 1; 1 100 1; 1 1 1];
convolved3x3 = conv2(binaryImage, mask3x3, 'same');
cleanedImage(convolved3x3 > 100) = 1;

% Step 3: Remove tiny spots (less than 3x3)
maskTiny = ones(3); % 3x3 mask
convolvedTiny = conv2(cleanedImage, maskTiny, 'same');
cleanedImage(convolvedTiny > 4.5) = 1; % Threshold 4.5

% Display results
figure;
subplot(1, 2, 1);
imshow(binaryImage, []);
title('Original Binary Image');

subplot(1, 2, 2);
imshow(cleanedImage, []);
title('Cleaned Image');

