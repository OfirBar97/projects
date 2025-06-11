% Load the image
I = imread('entire.bmp'); % Replace with your image file
grayImage = rgb2gray(I); % Convert to grayscale

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size JxJ
windowSize = 3;

% Compute After_low_I by averaging the surrounding {windowSize x windowSize} window
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

% Display the results
figure;
subplot(1, 3, 1);
imshow(grayImage);
title('Grayscale Image');

subplot(1, 3, 2);
imshow(uint8(After_low_I));
title('After Low-Pass Filtering');

subplot(1, 3, 3);
imshow(thresholdedImage, []);
title('Thresholded Image');
