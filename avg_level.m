function Average_level = avg_level(Loaded_image,windowSize)
% Creating Average level and Substractin grayscaled image by neighborhood average level

% Initialize variables
[rows, cols] = size(Loaded_image); % Size of the grayscale image
After_low_I = zeros(rows, cols); % Preallocate for efficiency

% Compute After_low_I by averaging the surrounding {windowSize x windowSize} window
for i = 1:rows
    for j = 1:cols
        % Determine the bounds of the window
        rStart = max(i - windowSize, 1);
        rEnd = min(i + windowSize, rows);
        cStart = max(j - windowSize, 1);
        cEnd = min(j + windowSize, cols);

        % Extract the neighborhood
        neighborhood = Loaded_image(rStart:rEnd, cStart:cEnd);

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
Average_level = Loaded_image- After_low_I;

% Display the difference image
subplot(3, 2, 3);
imshow(Average_level, []);
title('After Loaded_image - Average_level');
end