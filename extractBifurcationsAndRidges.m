function [map_of_bif, map_of_ridge] = extractBifurcationsAndRidges(thinnedImage)
    % This function returns the coordinates of bifurcations and ridges in matrix form.
    % Input:
    %   thinnedImage - binary skeletonized (thinned) image
    % Output:
    %   map_of_bif - matrix of bifurcation coordinates (Nx2 matrix)
    %   map_of_ridge - matrix of ridge coordinates (Nx2 matrix)
    
    % Detect bifurcations using the predefined masks
    bifurcations = detectBifurcations(thinnedImage);
    
    % Get coordinates of bifurcations
    [y_bif, x_bif] = find(bifurcations);
    
    % Create the map of bifurcation coordinates
    map_of_bif = [x_bif, y_bif]; % X is the column, Y is the row
    
    % Detect ridge segments (Assuming detectRidgeSegments function exists)
    [ridgeMap, ridgeCoords] = detectRidgeSegments(thinnedImage);
    
    % Create the map of ridge coordinates
    map_of_ridge = ridgeCoords; % Ridge coordinates already in Nx2 matrix format
    
end
