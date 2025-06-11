clc;
clear;

% Load the image
I = imread('fp3.bmp'); % Replace with your image file

grayImage = im2gray(I); % Convert to grayscale

subplot(3, 2, 1);
imshow(grayImage);
title('Original Gray Image');

%% 

% Creating Average level and Substractin grayscaled image by average level

% Initialize variables
[rows, cols] = size(grayImage); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Define the neighborhood size JxJ (windowSize is half the window width)
windowSize = 12;

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
subplot(3, 2, 2);
imshow(After_low_I);
title('After high filter I');

% Compute the difference image
differenceImage = grayImage- After_low_I;

% Display the difference image
subplot(3, 2, 3);
imshow(differenceImage, []);
title('After grayImage - After low I');

%%

% Contrast stretching of the 'difference image' using a linear
% transformation 

minIntensity = double(min(differenceImage(:)));
maxIntensity = double(max(differenceImage(:)));

% Perform linear contrast stretching
adjustedImage = (double(differenceImage) - minIntensity) * 3*(255 / (maxIntensity - minIntensity));

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
windowSize = 6; % Adjust as needed

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
            after_binary(i, j) = 0; % White for greater
        else
            after_binary(i, j) = 255;   % Black for smaller or equal
        end
    end
end

% Display the binary image
subplot(3, 2, 5);
imshow(after_binary);
title('Binary Image');




%%

% prunnig edges process



%%
% removing white single pixels and 2x2 white cubic pixels
%--------------------------------------------------------

% Binary image (1 for white, 0 for black)
binaryImage = after_binary > 0; % Ensure binary (logical) values

% Define a summing mask for the 3x3 neighborhood
maskA = [1 1 1; 1 1 1; 1 1 1]; % All 1s to sum the neighborhood

% Define a 4x4 mask for detecting 2x2 white blocks
mask_4x4 = ones(4); % 2x2 block of 1s

% looped 3 times to ensure riddens of 2x2 and 1x1 white pixels
for i= 1:3
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), maskA, 'same'); % Perform convolution
    
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
    
    % Use maskA again to turn the last pixels to black after maskB effect
    convResult = conv2(double(binaryImage), maskA, 'same');
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    binaryImage(singleWhitePixels) = 0;
end  

% Display results
figure;
subplot(1, 3, 2);
imshow(binaryImage);
title('Removed Single White Pixels and 2x2 White Blocks');

%%
%%%% remomving white pixels around 3x2 white pixels
%%%%
%%%% Input: binary image with white pixels (1) and black pixels (0)
%%%inputImage = binaryImage; % Replace with your binary image
%%%%
%%%% Define the mask for detecting 3x2 rectangles (with a padding frame)
%%%mask_3x2_custom = [1 1 1 1 1; 
%%%                   1 0 0 0 1; 
%%%                   1 0 0 0 1; 
%%%                   1 1 1 1 1];
%%%
%%%% Apply convolution/correlation to match the mask pattern
%%%convResult_3x2_custom = conv2(double(inputImage), double(mask_3x2_custom), 'same');
%%%%
%%%% Logical condition: Neighborhood sum equals the total number of 1s in the mask
%%%% (6 in this case, since there are six 1s in the mask)
%%%threeByTwoWhiteRectangles = (convResult_3x2_custom == sum(mask_3x2_custom(:))) & (inputImage == 1);
%%%%
%%%% Output: Keep only the detected 3x2 rectangles
%%%outputImage = zeros(size(inputImage)); % Initialize blank output
%%%outputImage(threeByTwoWhiteRectangles) = 1; % Mark detected 3x2 rectangles as white
%%%%
%%%% Display results
%%%figure;
%%%subplot(1, 2, 1);
%%%imshow(inputImage, []);
%%%title('Original Image');
%%%%
%%%subplot(1, 2, 2);
%%%imshow(outputImage, []);
%%%title('Detected 3x2 Rectangles');


%%


% removing black single pixels and 2x2 black cubic pixels
%--------------------------------------------------------

binaryImage= ~binaryImage; % inverting the image to use the same algo and invert back

% looped 3 times to ensure riddens of 2x2 and 1x1 white pixels
for i= 1:3
    % Apply convolution to compute neighborhood sums
    convResult = conv2(double(binaryImage), maskA, 'same'); % Perform convolution
    
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
    
    % Use maskA again to turn the last pixels to black after maskB effect
    convResult = conv2(double(binaryImage), maskA, 'same');
    singleWhitePixels = (convResult == 1) & (binaryImage == 1);
    binaryImage(singleWhitePixels) = 0;
end  


binaryImage = ~binaryImage;
% Display results

subplot(1, 3, 3);
imshow(binaryImage);
title('Black 1x1 & 2x2 spots cleared Image');

%%
%-------------------------------------------------------
% removing white 'edges'


% Define a 3x3 mask for detecting the neighborhood of each pixel
mask = ones(3); % 3x3 block of 1s

% Initialize the result image
resultImage = binaryImage;

% Apply convolution to compute neighborhood sums
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 5
% (i.e., the central pixel plus 4 surrounding white pixels)
singleWhitePixels = (convResult == 4) | (convResult == 5) & (binaryImage == 1);

% Remove the single Black pixels by setting them to White ~(1)
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
convResult = conv2(double(binaryImage), mask, 'same'); % Convolve the image with the mask

% Logical condition: Central pixel is white (1) and the total sum of the neighborhood is 4
% (i.e., the central pixel plus 3 surrounding white pixels)
singleBlackPixels = (convResult == 4) | (convResult == 5) & (binaryImage == 1);

% Remove the single white pixels by setting them to black (0)
resultImage(singleBlackPixels) = 0;
resultImage = ~resultImage; % using 'NOT' to change back the colors

% Display the results
subplot(1, 2, 2);
imshow(~resultImage);
title('Removed black Pixel edges (branches)');


%%
%thin_image=bwmorph(resultImage,'skel',inf);
%thin_image=~bwmorph(thin_image,'clean',inf);
%thin_image=bwmorph(thin_image,'spur',inf);
%figure;imshow(thin_image);title('spur Image');

%[h, w] = size(resultImage);
%temp_Im = ~resultImage;   
%Data
%r = 0;                          % Moving through Kernals
%count = 0;                      % Exiting criteria
%while 1 == 1
%    if r < 8                    % Looping through the structuring elements
%        r = r + 1;
%    else
%        r = 1;
%    end
%    strel1 = C(:,:,r);          % Structuring element
%    idx = idxset(r,:);          % Eliminate 'do not care' terms
%    
%    HnM = false(h,w);           % definition...
%    
%    for i = 2:h-1
%        for j = 2:w-1
%            el1 = temp_Im(i-1:i+1,j-1:j+1);  % Image element
%            if el1(5)                   % This is a gamble to save time...
%                if isequal(strel1(idx),el1(idx))    % Erosion
%                    HnM(i,j) = true;                % Save result
%                end
%            end
%        end
%    end
%
%    % Difference computation
%    difference = and(temp_Im,~HnM);     % Set difference (theory)
%    
%    if isequal(difference,temp_Im)      % No more changes
%        count = count + 1;              % Counter ++
%    else
%        count = 0;                      % Changes between previous and current - reset counter
%        temp_Im = difference;           % Replace previous with current
%        
%        % Output
%        % Comment to suppress animation (saves time)
%        imshow(temp_Im)
%        drawnow
%    end
%    if count == 8                       % All structuring elements have failed to make a change
%        output = temp_Im;
%        break                           % Stopping criteria
%    end
%end
%temp_Im = ~temp_Im;
%figure;
%subplot(1, 3, 1);
%imshow(temp_Im);
%title('skel Image');
%
subplot(1, 2, 1);
thin_image=~bwmorph(binaryImage,'clean',inf);
%thin_image=~bwmorph(thin_image,'thin',inf);
%thin_image=bwmorph(thin_image,'spur',inf);
thin_image=bwmorph(thin_image,'clean',inf);

imshow(thin_image);
title('cleaned Image');




subplot(1, 2, 2);
imshow(grayImage);
title('Original Gray Image');
%%  
