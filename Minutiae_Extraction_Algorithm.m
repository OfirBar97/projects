
% © 2025 Ofir Bar, Electrical Engineer.

% This MATLAB algorithm was developed as part of my graduation project. All rights reserved.
% For any questions, inquiries, or permissions, please contact me at: [ofirbar97@gmail.com].

% Unauthorized use, reproduction, or distribution of this work is strictly prohibited without prior written consent.

%%
clc;
clear;

% Load the image
I = imread('Finger_Print_Sample.bmp'); % image file

max_pixel = max(max(max(I)));
if max_pixel>255
    grayImage = im2gray(I); % Convert to grayscale
else 
    grayImage = rgb2gray(I);
end
grayImage=grayImage(255:891,142:582);  % dimensions focus
subplot(3, 2, 1);
imshow(grayImage);
title('Original Gray Image');

%% 

% Creating Average level and Substract grayscaled image by neighborhood average level

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_avg_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size JxJ 
windowSize =10;

% Compute After_high_I by averaging the surrounding {windowSize x windowSize} window
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
        After_avg_I(i, j) = mean(neighborhood(:));
    end
end

% Normalize the After_low_I to display properly
After_avg_I = uint8(After_avg_I);

% Display the smoothed image
subplot(3, 2, 2);
imshow(After_avg_I);
title('After high filter I');

% Compute the difference image
differenceImage = grayImage- After_avg_I;

% Display the difference image
subplot(3, 2, 3);
imshow(differenceImage, []);
title('After (grayImage - After_avg_I)');

%%

% Contrast stretching of the 'difference image' using a linear
% transformation 

minIntensity = double(min(differenceImage(:)));
maxIntensity = double(max(differenceImage(:)));

% Perform linear contrast stretching  % multiplied by 2
adjustedImage = (double(differenceImage) - minIntensity) *2*(255 / (maxIntensity - minIntensity));

% Convert back to uint8
adjustedImage = uint8(adjustedImage);

% Display the result
subplot(3, 2, 4);
imshow(adjustedImage);
title('Contrast Adjusted using Linear Stretch');

%%
% Thresholding function -> Binary image

% Initialize variables
[rows, cols] = size(adjustedImage);
after_binary = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size (i.e., window size)
windowSize =6; % Adjust as needed

% Perform thresholding based on mean of the surrounding window
for i = 1:rows
    for j = 1:cols
        % Determine the bounds of the window
        rStart = max(i - windowSize, 1);
        rEnd = min(i + windowSize, rows);
        cStart = max(j - windowSize, 1);
        cEnd = min(j + windowSize, cols);

        % Extract the neighborhood
        neighborhood = adjustedImage(rStart:rEnd, cStart:cEnd);

        % Compute the mean of the neighborhood
        meanValue = mean(neighborhood(:));

        % Assign pixel value in the new binary image
        if meanValue > adjustedImage(i, j)
            after_binary(i, j) = 0; % black
        else
            after_binary(i, j) = 255;   % white
        end
    end
end

% Display the binary image
subplot(3, 2, 5);
imshow(after_binary);
title('Binary Image');



%%

% prunnig edges process
%-------------------------------------------------------
% removing white 'edges'

after_binary = after_binary > 0; % ensuring binary representation

% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Initialize the result image
resultImage = after_binary;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(after_binary), mask, 'same'); 

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4 or 5
singleWhitePixels = ((convResult == 4)|(convResult == 5)) & (after_binary == 1);

% Remove the single white pixels by setting them to black 0
resultImage(singleWhitePixels) = 0;

% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Removed White Pixel edges (branches)');

%%
%-------------------------------------------------------
% removing black 'edges'

% Assuming binaryImage is the binary image (1 for white, 0 for black)
binaryImage = ~resultImage; % Example image

% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Initialize the result image
resultImage = binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same');

% Logical condition: Central pixel is white (1) and the total sum of the
% neighborhood is 4 or 5

singleBlackPixels  = ((convResult == 4) | (convResult == 5)) & (after_binary == 1);

% Remove the single white pixels by setting them to black (0)
resultImage(singleBlackPixels) = 0;
resultImage = ~resultImage; % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(resultImage);
title('Removed black Pixel edges (branches)');

%%
% removing white single pixels and 2x2 white cubic pixels
%--------------------------------------------------------

% Binary image (1 for white, 0 for black)
binaryImage = resultImage > 0; % Ensure binary (logical) values

% Define a 2x2 mask for detecting 1x1 white blocks
mask = ones(2);

% Define a 4x4 mask for detecting 2x2 white blocks
mask_4x4 = ones(4); 

% looped 2 times to ensure removal of 2x2 and 1x1 white pixels depends on
% sample pixel intensity 
for i= 1:2
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
   
    % Remove single white pixels by setting them to 0 in the binary image
    binaryImage(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (binaryImage == 1);
    
    % Remove 2x2 white blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  

% Display results
figure;
subplot(1, 2, 1);
imshow(binaryImage);
title('Removed Single White Pixels and 2x2 White Blocks');



%%


% removing black single pixels and 2x2 black cubic pixels
%--------------------------------------------------------

binaryImage= ~binaryImage; % inverting the image to use the same algo and invert back

% looped 2 times to ensure removal of 2x2 and 1x1 white pixels depends on
% sample pixel intensity 
for i= 1:2
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution
    
    % Logical condition: Neighborhood sum is 1, and the original pixel is 1
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    
    % Remove single white pixels by setting them to 0 in the binary image
    binaryImage(singleWhitePixels) = 0;
    
    % Apply convolution to compute 4x4 block sums
    convResult_4x4 = conv2(double(binaryImage), mask_4x4, 'same'); 
    
    % Logical condition: Neighborhood sum is 4 (indicating a 2x2 block of white pixels)
    twoByTwoWhitePixels = (convResult_4x4 <= 4)  & (binaryImage == 1);
    
    % Remove 2x2 black blocks by setting them to 0 in the binary image
    binaryImage(twoByTwoWhitePixels) = 0;
    
end  


binaryImage = ~binaryImage;

% Display results
subplot(1, 2,2);
imshow(binaryImage);
title('Black 1x1 & 2x2 spots cleared Image');

%%
%-------------------------------------------------------
% removing white 'edges'

binaryImage = binaryImage > 0; % ensuring binary representation

% Initialize the result image
resultImage = binaryImage;

% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s


% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); 

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4 or 5
singleWhitePixels = ((convResult == 4)|(convResult == 5)) & (binaryImage == 1);

% Remove the single white pixels by setting them to black 0
resultImage(singleWhitePixels) = 0;

% Display the results
figure;

subplot(1, 2, 1);
imshow(resultImage);
title('Removed White Pixel edges (branches)');

%%
%-------------------------------------------------------
% removing black 'edges'

binaryImage = ~resultImage; % inverting

% Initialize the result image
resultImage = binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); 

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4 or 5
singleWhitePixels = ((convResult == 4)|(convResult == 5)) & (binaryImage == 1);

% Remove the single white pixels by setting them to black 0
resultImage(singleWhitePixels) = 0;
resultImage = ~resultImage; % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(resultImage);
title('Removed black Pixel edges (branches)');
%%
% checking for last 1x1 noise pixels ( if theres any and if so removes them)

binaryImage=resultImage;
% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Perform convolution

% Logical condition: Neighborhood sum is 1, and the original pixel is 1
singleWhitePixels = (convResult == 1) & (binaryImage == 1);

% Remove single white pixels by setting them to 0 in the binary image
binaryImage(singleWhitePixels) = 0;

% using NOT to switch colors and use the same algo
binaryImage= ~binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same');

% Logical condition: Neighborhood sum is 1, and the original pixel is 1
singleWhitePixels = (convResult == 1) & (binaryImage == 1);

% Remove single white pixels by setting them to 0 in the binary image
binaryImage(singleWhitePixels) = 0;


imshow(~binaryImage);
title('Second Black 1x1 & White 1x1 spots cleared Image');
%%
% thinning process

thin_image=binaryImage;

subplot(1, 2, 1);

thin_image=~bwmorph(thin_image,'thin',inf);
thin_image=bwmorph(thin_image,'spur',inf);

thin_image=bwareaopen(~thin_image,3);
thin_image=~thin_image;
%subplot(1, 2, 1);
imshow(thin_image);
title('Thinned Image');

subplot(1, 2, 2);
imshow(grayImage);
title('Original Gray Image');

%%  

% Minutiae extraction

s = size(thin_image); % Get the size of the thinned binary image
N = 3; % Window size (3x3 neighborhood for minutiae detection)
mat = ones(size(thin_image)); % Temporary matrix for storing local neighborhoods
n = (N-1)/2; % Half window size
r = s(1) + 2*n; % New number of rows after padding
c = s(2) + 2*n; % New number of columns after padding

% Initialize temporary padded image and output matrices
double temp(r, c); % Declare temporary image matrix
temp = zeros(r, c); % Initialize with zeros (padding)
bifurcation = zeros(r, c); % Matrix to store bifurcation points
ridge = zeros(r, c); % Matrix to store ridge end points

% Copy the original image into the center of the padded matrix
temp((n+1):(end-n), (n+1):(end-n)) = thin_image(:,:);

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
plot(NaN, NaN, 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % green square for ridge endings
plot(NaN, NaN, 'sb', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); % Blue square for bifurcations
legend({'Ridges End', 'Bifurcation'}, 'Location', 'northeast', 'FontSize', 10);
hold off;

subplot(1,2,2)
imshow(grayImage);
title('Original Gray Image');
