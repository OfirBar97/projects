function [roi_mask, fingerprint_roi, filtered_minutiae] = extract_fingerprint_roi(adjustedImage, minutiae)
% EXTRACT_FINGERPRINT_ROI - Extracts the fingerprint ROI from an image and suppresses background
%
% Inputs:
%   adjustedImage - Grayscale fingerprint image (e.g., 300x375)
%   minutiae      - (Optional) Nx2 matrix of [x, y] minutiae coordinates
%
% Outputs:
%   roi_mask         - Binary mask for ROI
%   fingerprint_roi  - Image with background removed
%   filtered_minutiae - (Optional) Minutiae within ROI only

    % --- Step 1: Binarization and Thinning ---
    bw = imbinarize(adjustedImage, 'adaptive', 'Sensitivity', 0.55 );
    thin_bw = bwmorph(bw, 'thin', Inf);

    % --- Step 2: Morphological Closing ---
    se_close = strel('disk', 6);
    closed = imclose(thin_bw, se_close);

    % --- Step 3: Fill Holes ---
    filled = imfill(closed, 'holes');

    % --- Step 4: Opening to remove small noise ---
    se_open = strel('disk', 3);
    opened = imopen(filled, se_open);

    % --- Step 5: Remove small objects ---
    clean = bwareaopen(opened, 500);

    % --- Step 6: Erode to clean border ---
    roi_mask = imerode(clean, strel('disk', 12));

    % --- Step 7: Apply mask to original image ---
    fingerprint_roi = adjustedImage;
    fingerprint_roi(~roi_mask) = 255;

    % --- Step 8: (Optional) Filter minutiae by ROI ---
    if nargin == 2 && ~isempty(minutiae)
    % Round and clip coordinates to image bounds
    minutiae = round(minutiae);
    [H, W] = size(roi_mask);
    in_bounds = minutiae(:,1) >= 1 & minutiae(:,1) <= W & ...
                minutiae(:,2) >= 1 & minutiae(:,2) <= H;

    valid_minutiae = minutiae(in_bounds, :);
    inds = sub2ind(size(roi_mask), valid_minutiae(:,2), valid_minutiae(:,1));
    mask_values = roi_mask(inds);
    filtered_minutiae = valid_minutiae(mask_values == 1, :);
else
    filtered_minutiae = [];
end