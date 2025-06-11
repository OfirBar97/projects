function cleanedImage = removePatternPixels(thinnedImage)
% Remove specific patterns by convolving and discarding the central pixel
% thinnedImage: binary image after thinning (0 background, 1 ridges)

% Ensure binary
thinnedImage = thinnedImage > 0;
cleanedImage = thinnedImage;

% Define the 6 masks
masks = {
    [0 0 0; 0 1 0; 1 1 0], ...
    [0 0 0; 0 1 0; 0 1 1], ...
    [1 1 0; 0 1 0; 0 0 0], ...
    [0 1 1; 0 1 0; 0 0 0], ...
    [0 0 0; 1 1 0; 1 0 0], ...
    [0 0 1; 0 1 1; 0 0 0]
};

% Loop through all masks
for k = 1:length(masks)
    currentMask = masks{k};
    % Slide a 3x3 window over the image
    for i = 2:size(thinnedImage, 1)-1
        for j = 2:size(thinnedImage, 2)-1
            window = cleanedImage(i-1:i+1, j-1:j+1);
            if isequal(window, currentMask)
                cleanedImage(i, j) = 0; % discard center pixel
            end
        end
    end
end
end

