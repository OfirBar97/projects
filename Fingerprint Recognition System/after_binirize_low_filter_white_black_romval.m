% Load the image
I = imread('entire.bmp'); % Replace with your image file
grayImage = rgb2gray(I); % Convert to grayscale

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size
windowSize = 1;

% Compute After_low_I by averaging the surrounding 10x10 window
for i = 1:rows
    for j = 1:cols
        % Determine the bounds of the window
        rStart = max(i - windowSize, 1);
        rEnd = min(i + windowSize, rows);
        cStart = max(j - windowSize, 1);
        cEnd = min(j + windowSize, cols);

        % Extract the neighborhood
        neighborhood = grayImage(rStart:rEnd, cStart:cEnd);

        % Compute the average of the neighborhood
        After_low_I(i, j) = mean(neighborhood(:));
    end
end

% Initialize the new thresholded image
thresholdedImage = zeros(rows, cols);

% Apply the thresholding operation
for i = 1:rows
    for j = 1:cols
        % Determine the bounds of the window
        rStart = max(i - windowSize, 1);
        rEnd = min(i + windowSize, rows);
        cStart = max(j - windowSize, 1);
        cEnd = min(j + windowSize, cols);

        % Extract the neighborhood
        neighborhood = After_low_I(rStart:rEnd, cStart:cEnd);

        % Compute the average of the neighborhood
        neighborhoodMean = mean(neighborhood(:));

        % Apply the thresholding condition
        if After_low_I(i, j) > neighborhoodMean
            thresholdedImage(i, j) = 255;
        else
            thresholdedImage(i, j) = 0;
        end
    end
end
binaryImage =thresholdedImage;
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

% Display the results
figure;
subplot(2, 2, 1);
imshow(binaryImage);
title('Original Binary Image');

subplot(2, 2, 2);
imshow(cleanedImage);
title('Cleaned Image-after white');


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
if cleanedImage(convolvedTiny) > 4.5 % Threshold 4.5
    cleanedImage(convolvedTiny)= 1;
end

% Display results

subplot(2, 2, 3);
imshow(binaryImage, []);
title('Original Binary Image');

subplot(2, 2, 4);
imshow(cleanedImage, []);
title('Cleaned Image-after black');