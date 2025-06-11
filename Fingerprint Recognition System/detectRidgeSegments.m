function [ridgeMap, ridgeCoords] = detectRidgeSegments(thinnedImage)
% Detect ridge segments using 3x3 masks with 2 connected '1's
% Inputs:
%   thinnedImage - binary image after thinning (1 = ridge, 0 = background)
% Outputs:
%   ridgeMap - binary map of detected ridge segments
%   ridgeCoords - Nx2 matrix with [row, col] indices of ridge segments

% Ensure binary
thinnedImage = thinnedImage > 0;

% Output initialization
ridgeMap = false(size(thinnedImage));
ridgeCoords = [];

% Define all 3x3 masks with 2 connected '1's
masks = {
    [0 0 0; 0 1 1; 0 0 0], ... % Horizontal
    [0 1 0; 0 1 0; 0 0 0], ... % Vertical
    [0 0 0; 0 1 0; 0 1 0], ... % Vertical (inverted)
    [0 0 0; 1 1 0; 0 0 0], ... % Horizontal (inverted)
    [0 0 0; 1 1 0; 0 0 0], ... % Diagonal 
    [1 0 0; 0 1 0; 0 0 0], ... % Diagonal upper left to center
    [0 0 1; 0 1 0; 0 0 0], ... % Diagonal upper right to center
    [0 0 0; 0 1 0; 1 0 0], ... % Diagonal center to lower left
    [0 0 0; 0 1 0; 0 0 1]  ... % Diagonal center to lower right
};

% Slide 3x3 window
for i = 2:size(thinnedImage, 1)-1
    for j = 2:size(thinnedImage, 2)-1
        window = thinnedImage(i-1:i+1, j-1:j+1);
        if sum(window(:)) == 2
            for k = 1:length(masks)
                if isequal(window, masks{k})
                    ridgeMap(i, j) = true;
                    ridgeCoords = [ridgeCoords; i, j];
                    break;
                end
            end
        end
    end
end
end
