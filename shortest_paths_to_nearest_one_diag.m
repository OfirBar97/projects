function distances = shortest_paths_to_nearest_one_diag(binaryMatrix)
% God Bless Breadth First Search Algorithm
    [rows, cols] = size(binaryMatrix);
    [onesY, onesX] = find(binaryMatrix);  % coordinates of 1s
    numOnes = length(onesX);
    distances = inf(numOnes, 1);

    % Define 8-connected directions
    directions = [ -1 -1; -1 0; -1 1;
                    0 -1;        0 1;
                    1 -1;  1 0;  1 1 ];

    for idx = 1:numOnes
        visited = false(rows, cols);
        queue = [onesY(idx), onesX(idx), 0];  % [y, x, distance]
        visited(onesY(idx), onesX(idx)) = true;

        found = false;

        while ~isempty(queue)
            [y, x, dist] = deal(queue(1,1), queue(1,2), queue(1,3));
            queue(1, :) = [];  % dequeue

            for d = 1:size(directions, 1)
                ny = y + directions(d, 1);
                nx = x + directions(d, 2);

                if ny >= 1 && ny <= rows && nx >= 1 && nx <= cols && ~visited(ny, nx)
                    visited(ny, nx) = true;
                    if binaryMatrix(ny, nx) == 1 && ~(ny == onesY(idx) && nx == onesX(idx))
                        distances(idx) = dist + 1;
                        found = true;
                        break;  % Found nearest other 1
                    else
                        queue(end+1, :) = [ny, nx, dist+1];  % enqueue
                    end
                end
            end

            if found
                break;
            end
        end
    end
end
