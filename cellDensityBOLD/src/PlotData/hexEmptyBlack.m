function [color_hex_triplet] = hexEmptyBlack(color_hex_triplet)
    for i = 1:length(color_hex_triplet)
        if isempty(color_hex_triplet{i})
            color_hex_triplet(i) = {'000000'};
        end
    end
end

