function map = create_map_from_coordinates(coordinates, map_size)
    % Initialize the map with zeros
    map = zeros(map_size);

    % Loop through the list of coordinates and set corresponding positions to 1
    for i = 1:size(coordinates, 1)
        x = coordinates(i, 1);
        y = coordinates(i, 2);
        
        % Check if the coordinates are within the bounds of the map
        if x <= map_size(1) && y <= map_size(2)
            map(x, y) = 1;  % Set the position at (x, y) to 1
        end
    end
end
