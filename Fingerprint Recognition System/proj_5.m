%% Load the image
clc;
clear;

grayImage = imread('ofir_7_1.bmp'); %  image file
filename = sprintf('map_of_ones_raz1.png'); % save file name as

figure;
subplot(2, 2, 1);
imshow(grayImage);
title('Original Gray Image');

% finding ROI

% ROI function
[roi_mask, img_roi] = extract_fingerprint_roi(grayImage);

% Show result
imshow(img_roi); title('Fingerprint ROI');

grayImage=img_roi;

%%%[rows, cols] = size(grayImage);
%%%
%%%xCenter = round(cols / 2);  % x axis (columns)
%%%yCenter = round(rows / 2);  % y axis (rows)
%%%
%%%Zoom_windowSize = 125;  % amount to zoom in
%%%
%%%% Define bounds for cropping
%%%xStart = max(1, xCenter - Zoom_windowSize);
%%%xEnd   = min(cols, xCenter + Zoom_windowSize);
%%%yStart = max(1, yCenter - Zoom_windowSize);
%%%yEnd   = min(rows, yCenter + Zoom_windowSize);

% Crop the image around the center
%%%grayImage = grayImage(yStart:yEnd, xStart:xEnd);  % rows (y), cols (x)
imshow(grayImage);

%% Creating Average level and Substractin grayscaled image by neighborhood average level

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size  (windowSize is 'windowSize +1')
windowSize = 8;

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

% Normalize the After_low_I to display properly
After_low_I = uint8(After_low_I);

% Display the smoothed image
subplot(2, 2, 2);
imshow(After_low_I);
title('After high filter I');

% Compute the difference image
differenceImage = grayImage- After_low_I;

% Display the difference image
subplot(2, 2, 3);
imshow(differenceImage, []);
title('After grayImage - After High I');

%% Contrast stretching of the 'difference image' using a linear transformation 

minIntensity = double(min(differenceImage(:)));
maxIntensity = double(max(differenceImage(:)));

% Perform linear contrast stretching
adjustedImage = (double(differenceImage) - minIntensity) *2*(255 / (maxIntensity - minIntensity));

% Convert back to uint8
adjustedImage = uint8(adjustedImage);

% Display the result
subplot(2, 2, 4);
imshow(adjustedImage);
title('Contrast Adjusted using Linear Stretch');

%% Thresholding using Niblack method
windowSize = 5; % Local neighborhood size
k = -0.1;       % Niblack's parameter (typically negative)

% Convert image to double
img = double(adjustedImage);

% Create box filter for local mean
filterKernel = ones(windowSize, windowSize) / (windowSize^2);

% Compute local mean
meanImage = conv2(img, filterKernel, 'same');

% Compute local standard deviation
squaredImage = img.^2;
meanSquared = conv2(squaredImage, filterKernel, 'same');
stdImage = sqrt(meanSquared - meanImage.^2);

% Niblack threshold
threshold = meanImage + k * stdImage;

% Binarize image
after_binary = img > threshold;

% Optional post-processing
after_binary = ~after_binary;
after_binary = ~bwmorph(after_binary, 'fill', Inf);

% Display result
imshow(after_binary);
title('Niblack Binary Image');

%%
% removing white single pixels and 2x2 white cubic pixels
%--------------------------------------------------------
binaryImage = after_binary; % image 

% Define a 4x4 mask for detecting 2x2 white blocks
mask_4x4 = ones(4); % 4x4 block of 1s
mask =ones(3);
% looped 3 times to ensure removal of 2x2 and 1x1 white pixels
for i= 1:4
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    binaryImage(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (binaryImage == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 white blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  

% Display results
figure;
subplot(1, 2, 1);
imshow(binaryImage);
title('Removed Single White Pixels and 2x2 White Blocks');

%% removing black single pixels and 2x2 black cubic pixels
% 
%--------------------------------------------------------

binaryImage= ~binaryImage; % inverting the image to use the same algo and invert back

% looped 3 times to ensure riddens of 2x2 and 1x1 white pixels
for i= 1:2
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    binaryImage(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (binaryImage == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  
after_binary= ~binaryImage; % inverting the image to use the same algo and invert back


% Display results

subplot(1, 2,2);
imshow(after_binary);
title('Removed Single Black Pixels and 2x2 Black Blocks');

%% prunnig edges process
%-------------------------------------------------------
% removing black 'edges'

after_binary = after_binary > 0; % making sure '0' or '1's
% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s
for j=1:3
% Initialize the result image
resultImage = after_binary;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(after_binary), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhitePixels = ((convResult == 4)) & (after_binary == 1);

% Remove the single Black pixels by setting them to White 1
resultImage(singleWhitePixels) = 0;
end

% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Removed black  (branches)');
%%
%-------------------------------------------------------
% removing white 'edges'

% Assuming binaryImage is the binary image (1 for white, 0 for black)
binaryImage = resultImage; % Example image

% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4
% (i.e., the central pixel plus 3 surrounding white pixels)
singleBlackPixels  = (convResult == 4) & (after_binary == 1);

% Remove the single white pixels by setting them to black (0)
resultImage(singleBlackPixels) = 0;
%resultImage = ~resultImage; % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(resultImage);
title('Removed white  (branches)');
%% Thinning Process

figure;
subplot(1, 2, 1);
%thin_image=bwmorph(~binaryImage,'clean',inf);
thin_image=~bwmorph(~resultImage,'thin',inf);
thin_image=~bwmorph(thin_image,'spur',inf);

thin_image=~thin_image;

imshow(thin_image);
title('Thinned Image');

subplot(1, 2, 2);
imshow(grayImage);
title('Original Gray Image');

%% Removing Branches Created by the Thinning process
%--------------------------------------------------------

thin_image_2= ~thin_image; % inverting the image to use the same algo and invert back

% discards 2x2 and 1x1 white pixels
for i= 1:5
    % Apply convolution to compute neighborhood sums
    convResult_2 = conv2(double(thin_image_2), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult_2 == 2) & (thin_image_2 == 1);
    
    % Output: Highlight or process single white pixels as needed
    % Remove single white pixels by setting them to 0 in the binary image
    thin_image_2(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(thin_image_2), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 2)  & (thin_image_2 == 1);
    
    % Output: Highlight or process 4x4 white blocks as needed
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  

thin_image_2 = ~thin_image_2;
% Display results
figure
imshow(thin_image_2);
title('Thinned Image v2');

%%  

% Minutiae extraction
s = size(thin_image_2); % Get the size of the thinned binary image
N = 3; % Window size (3x3 neighborhood for minutiae detection)
mat = ones(size(thin_image_2)); % Temporary matrix for storing local neighborhoods
n = (N-1)/2; % Half window size
r = s(1) + 2*n; % New number of rows after padding
c = s(2) + 2*n; % New number of columns after padding

% Initialize temporary padded image and output matrices
double temp(r, c); % Declare temporary image matrix
temp = zeros(r, c); % Initialize with zeros (padding)
bifurcation = zeros(r, c); % Matrix to store bifurcation points
ridge = zeros(r, c); % Matrix to store ridge end points

% Copy the original image into the center of the padded matrix
temp((n+1):(end-n), (n+1):(end-n)) = thin_image_2(:,:);

% Create an RGB representation of the padded image for visualization
thinned_Img = zeros(r, c, 3); % Initialize 3-channel (RGB) image
thinned_Img(:,:,1) = temp .* 255; % Set red channel
thinned_Img(:,:,2) = temp .* 255; % Set green channel
thinned_Img(:,:,3) = temp .* 255; % Set blue channel

% Sliding window for minutiae detection
for x = (n+1+10):(s(1)+n-10) % Avoid processing near the padded border
    for y = (n+1+10):(s(2)+n-10)
        e = 1; % Row index for 3x3 window
        for k = x-n:x+n
            f = 1; % Column index for 3x3 window
            for l = y-n:y+n
                mat(e, f) = temp(k, l); % Extract 3x3 neighborhood
                f = f + 1;
            end
            e = e + 1;
        end
        % Check if the central pixel is black (ridge pixel)
        if(mat(2, 2) == 0)
            ridge(x, y) = sum(sum(~mat)); % Count white neighbors for ridge endings
            bifurcation(x, y) = sum(sum(~mat)); % Count white neighbors for bifurcations
        end
    end
end

% RIDGE END FINDING
[ridge_x, ridge_y] = find(ridge == 2); % Find coordinates of ridge endings
len = length(ridge_x); % Number of ridge endings
% For Display (mark ridge endings with green squares)
for i = 1:len
    % Clear area around ridge ending (set to black)
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)-3), 1:3) = 0;
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)+3), 1:3) = 0;
    thinned_Img((ridge_x(i)-3), (ridge_y(i)-3):(ridge_y(i)+3), 1:3) = 0;
    thinned_Img((ridge_x(i)+3), (ridge_y(i)-3):(ridge_y(i)+3), 1:3) = 0;

    % Mark ridge ending in green
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)-3), 2) = 255;
    thinned_Img((ridge_x(i)-3):(ridge_x(i)+3), (ridge_y(i)+3), 2) = 255;
    thinned_Img((ridge_x(i)-3), (ridge_y(i)-3):(ridge_y(i)+3), 2) = 255;
    thinned_Img((ridge_x(i)+3), (ridge_y(i)-3):(ridge_y(i)+3), 2) = 255;
end

% BIFURCATION FINDING
[bifurcation_x, bifurcation_y] = find(bifurcation == 4); % Find coordinates of bifurcations
len = length(bifurcation_x); % Number of bifurcations
% For Display (mark bifurcations with blue squares)
for i = 1:len
    % Clear area around bifurcation (set to black)
    thinned_Img((bifurcation_x(i)-3):(bifurcation_x(i)+3), (bifurcation_y(i)-3), 1:2) = 0;
    thinned_Img((bifurcation_x(i)-3):(bifurcation_x(i)+3), (bifurcation_y(i)+3), 1:2) = 0;
    thinned_Img((bifurcation_x(i)-3), (bifurcation_y(i)-3):(bifurcation_y(i)+3), 1:2) = 0;
    thinned_Img((bifurcation_x(i)+3), (bifurcation_y(i)-3):(bifurcation_y(i)+3), 1:2) = 0;
end

figure;
subplot(1,2,1)
imshow(thinned_Img);
title('Minutiae');
hold on;
% Dummy markers for legend
plot(NaN, NaN, 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Red square for ridge endings
plot(NaN, NaN, 'sb', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % Blue square for bifurcations
legend({'Ridges End', 'Bifurcation'}, 'Location', 'northeast', 'FontSize', 10);
hold off;

subplot(1,2,2)
imshow(grayImage);
title('Original Gray Image');

%% creating new image 'result' that contains ridges end and bifurcation, pre-filtered.

% Get image size
[rows, cols] = size(thin_image_2);

% Initialize result matrix with zeros
result = zeros(rows, cols);

% Loop through the image, avoiding the borders
for i = 2:rows-10
    for j = 2:cols-10
        if thin_image_2(i, j) == 0  % Check only ridge points
            % Extract 3x3 neighborhood
            neighborhood = thin_image_2(i-1:i+1, j-1:j+1);
            
            % Count black pixels (zeros)
            black_pixel_count = sum(neighborhood(:));
            
            % Apply the conditions
            if black_pixel_count == 7
                result(i, j) = 1;  % Ridge ending
            end
                
                
            if black_pixel_count ==  5
                result(i, j) = 1;  % Bifurcation
            end
            
        end
    end
end
figure;

%% filtering neighboring intreset points to singles

[rows, cols] = size(result);
filtered_result = result; % Copy the original image

% Loop through the image with a 3x3 sliding window
for i = 10:rows-1
    for j = 10:cols-1
        % Extract the 3x3 window
        window = result(i-1:i+1, j-1:j+1);
        
        % Check if there are exactly 3 ones in the window
        if sum(window(:)) == 3
            % Reduce to a single '1' (choose priority: center, top-left, or random)
            newWindow = zeros(3,3); % Reset the 3x3 window to all zeros
            
            if window(2,2) == 1
                newWindow(2,2) = 1; % Keep center pixel if it exists
            else
                indices = find(window == 1); % Find indices of '1' pixels
                keep_idx = indices(1); % Keep the first found (top-left priority)
                newWindow(keep_idx) = 1;
            end
            
            % Update the new image with the modified window
            filtered_result(i-1:i+1, j-1:j+1) = newWindow;
        end
    end
end

number_of_points = sum(filtered_result,"all");
% Extract points from newImage (assuming it contains binary minutiae points)
[y, x] = find(filtered_result); % Get coordinates of detected points

figure; % Create a new figure

% First subplot: Original Gray Image
subplot(1,2,1);
imshow(filtered_result); % Display the original gray image
title('Original Gray Image');
hold on; % Keep the image while plotting additional elements

% Plot detected points on the gray image
plot(x, y, 'go', 'MarkerSize', 8, 'LineWidth', 1); % Green circles
legend('Points of Interest'); % Correct legend syntax

% Second subplot: Thinned Binary Image
subplot(1,2,2);
imshow(thin_image_2); % Display the binary image
title('Thinned Image');
hold on; % Keep the image for overlaying points

% Plot the same detected points on the thinned binary image
plot(x, y, 'bo', 'MarkerSize', 8, 'LineWidth', 1); % Green circles
legend('Minutiae'); % Correct legend syntax
hold off; % Release hold for this subplot

number_of_points;

imwrite(filtered_result, filename);
