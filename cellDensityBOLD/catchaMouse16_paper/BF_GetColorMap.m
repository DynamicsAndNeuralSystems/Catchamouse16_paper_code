function cmapOut = BF_GetColorMap(whichMap,numGrads,cellOut,flipMe)
% BF_GetColorMap    Colorbrewer colors for Matlab.
%
% Returns a nice set of colors to make a nice colormap using the color schemes
% from http://colorbrewer2.org/
%
% The online tool, colorbrewer2, is copyright Cynthia Brewer, Mark Harrower and
% The Pennsylvania State University.
%
%---INPUTS:
%
% whichMap, the name of a colormap (see long list below)
%
% numGrads, the number of colors to return from that color scheme (some maps can
%           support larger numbers of colors, and the minimum is usually 3)
%
% cellOut, (i) 1: returns a cell of colors, where each component is an rgb
%                 3-vector
%          (ii) 0: returns a numGrads x 3 matrix for use in the Matlab colormap
%                  function, for example
% flipMe, (i) 1: inverts the ordering of the colors.
%            (ii) 0: doesn't invert the ordering of the colors.
%
%---EXAMPLE USAGE:
%
% 1. Set the colormap to the redyellowblue colormap with 8 gradations:
% colormap(BF_GetColorMap('redyellowblue',8));
%
% 2. Set blue/red maps for [-1,+1] bounded data:
% caxis([-1,1])
% colormap([flipud(BF_GetColorMap('blues',9));1,1,1;BF_GetColorMap('reds',9)])
%
% 3. Get colors for lines and plot data stored in 5 rows of matrix y:
% myColors = BF_GetColorMap('accent',5); hold on
% for i = 1:5,
%     plot(y(i,:),'color',myColors(i,:));
% end

% ------------------------------------------------------------------------------
% Copyright (C) 2020, Ben D. Fulcher <ben.d.fulcher@gmail.com>,
% <http://www.benfulcher.com>
%
% If you use this code for your research, please cite the following two papers:
%
% (1) B.D. Fulcher and N.S. Jones, "hctsa: A Computational Framework for Automated
% Time-Series Phenotyping Using Massive Feature Extraction, Cell Systems 5: 527 (2017).
% DOI: 10.1016/j.cels.2017.10.001
%
% (2) B.D. Fulcher, M.A. Little, N.S. Jones, "Highly comparative time-series
% analysis: the empirical structure of time series and their methods",
% J. Roy. Soc. Interface 10(83) 20130048 (2013).
% DOI: 10.1098/rsif.2013.0048
%
% This function is free software: you can redistribute it and/or modify it under
% the terms of the GNU General Public License as published by the Free Software
% Foundation, either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
% details.
%
% You should have received a copy of the GNU General Public License along with
% this program. If not, see <http://www.gnu.org/licenses/>.
% ------------------------------------------------------------------------------

%-------------------------------------------------------------------------------
% Check inputs and set defaults:
%-------------------------------------------------------------------------------
% Number of gradations, numGrads:
if nargin < 2 || isempty(numGrads)
    numGrads = 6;
end
if nargin < 3 || isempty(cellOut)
    cellOut = false; % output as matrix for colourmap instead
end
if nargin < 4 || isempty(flipMe)
    flipMe = false; % flips order of output
end

% Minimum number of groups is 3 for some color maps:
numGrads0 = numGrads;
if numGrads < 3
    numGrads = 3;
end

% Choices for whichMap:
%
% SEQUENTIAL MONOHUE:
% 'blues'
% 'greens'
% 'oranges'
% 'purples'
% 'reds'
%
% SEQUENTIAL MULTIHUE:
% 'bluegreen'
% 'bluepurple'
% 'greenblue'
% 'orangered'
% 'purpleblue'
% 'purplebluegreen'
% 'purplered'
% 'redpurple'
% 'yellowgreen'
% 'yellowgreenblue'
% 'yelloworangebrown'
% 'yelloworangered'
%
% DIVERGENT:
% 'spectral'
% 'redyellowgreen'
% 'redyellowblue'
% 'redgray'
% 'redblue'
% 'purpleorange'
% 'purplegreen'
% 'pinkgreen'
% 'browngreen'
%
% QUALITATIVE:
% 'accent'
% 'dark2'
% 'paired'
% 'pastel1'
% 'pastel2'
% 'set1'
% 'set2'
% 'set3'

%-------------------------------------------------------------------------------
% Match the specified color map name and output the required RGB matrix
%-------------------------------------------------------------------------------

switch whichMap
    case 'blues'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [222, 235, 247;
                    158, 202, 225;
                    49, 130, 189];
            case 4
                cmapOut = [239, 243, 255;
                    189, 215, 231;
                    107, 174, 214;
                    33, 113, 181];
            case 5
                cmapOut = [239, 243, 255;
                    189, 215, 231;
                    107, 174, 214;
                    49, 130, 189;
                    8, 81, 156];
            case 6
                cmapOut = [239, 243, 255;
                    198, 219, 239;
                    158, 202, 225;
                    107, 174, 214;
                    49, 130, 189;
                    8, 81, 156];
            case 7
                cmapOut = [239, 243, 255;
                    198, 219, 239;
                    158, 202, 225;
                    107, 174, 214;
                    66, 146, 198;
                    33, 113, 181;
                    8, 69, 148];
            case 8
                cmapOut = [247, 251, 255;
                    222, 235, 247;
                    198, 219, 239;
                    58, 202, 225;
                    107, 174, 214;
                    66, 146, 198;
                    33, 113, 181;
                    8, 69, 148];
            case 9
                cmapOut = [247, 251, 255;
                    222, 235, 247;
                    198, 219, 239;
                    158, 202, 225;
                    107, 174, 214;
                    66, 146, 198;
                    33, 113, 181;
                    8, 81, 156;
                    8, 48, 107];
        end
    case 'greens'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [229, 245, 224;
                    161, 217, 155;
                    49, 163, 84];
            case 4
                cmapOut = [237, 248, 233;
                    186, 228, 179;
                    116, 196, 118;
                    35, 139, 69];
            case 5
                cmapOut = [237, 248, 233;
                    186, 228, 179;
                    116, 196, 118;
                    49, 163, 84;
                    0, 109, 44];
            case 6
                cmapOut = [237, 248, 233;
                    199, 233, 192;
                    161, 217, 155;
                    116, 196, 118;
                    49, 163, 84;
                    0, 109, 44];
            case 7
                cmapOut = [237, 248, 233;
                    199, 233, 192;
                    161, 217, 155;
                    116, 196, 118;
                    65, 171, 93;
                    35, 139, 69;
                    0, 90, 50];
            case 8
                cmapOut = [247, 252, 245;
                    229, 245, 224;
                    199, 233, 192;
                    161, 217, 155;
                    116, 196, 118;
                    65, 171, 93;
                    35, 139, 69;
                    0, 90, 50];
            case 9
                cmapOut = [247, 252, 245;
                    229, 245, 224;
                    199, 233, 192;
                    161, 217, 155;
                    116, 196, 118;
                    65, 171, 93;
                    35, 139, 69;
                    0, 109, 44;
                    0, 68, 27];
        end
    case 'oranges'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [254, 230, 206;
                    253, 174, 107;
                    230, 85, 13];
            case 4
                cmapOut = [254, 237, 222;
                    253, 190, 133;
                    253, 141, 60;
                    217, 71, 1];
            case 5
                cmapOut = [254, 237, 222;
                    253, 190, 133;
                    253, 141, 60;
                    230, 85, 13;
                    166, 54, 3];
            case 6
                cmapOut = [254, 237, 222;
                    253, 208, 162;
                    253, 174, 107;
                    253, 141, 60;
                    230, 85, 13;
                    166, 54, 3];
            case 7
                cmapOut = [254, 237, 222;
                    253, 208, 162;
                    253, 174, 107;
                    253, 141, 60;
                    241, 105, 19;
                    217, 72, 1;
                    140, 45, 4];
            case 8
                cmapOut = [255, 245, 235;
                    254, 230, 206;
                    253, 208, 162;
                    253, 174, 107;
                    253, 141, 60;
                    241, 105, 19;
                    217, 72, 1;
                    140, 45, 4];
            case 9
                cmapOut = [255, 245, 235;
                    254, 230, 206;
                    253, 208, 162;
                    253, 174, 107;
                    253, 141, 60;
                    241, 105, 19;
                    217, 72, 1;
                    166, 54, 3;
                    127, 39, 4];
        end
    case 'purples'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [239, 237, 245;
                    188, 189, 220;
                    117, 107, 177];
            case 4
                cmapOut = [242, 240, 247;
                    203, 201, 226;
                    158, 154, 200;
                    106, 81, 163];
            case 5
                cmapOut = [242, 240, 247;
                    203, 201, 226;
                    158, 154, 200;
                    117, 107, 177;
                    84, 39, 143];
            case 6
                cmapOut = [242, 240, 247;
                    218, 218, 235;
                    188, 189, 220;
                    158, 154, 200;
                    117, 107, 177;
                    84, 39, 143];
            case 7
                cmapOut = [242, 240, 247;
                    218, 218, 235;
                    188, 189, 220;
                    158, 154, 200;
                    128, 125, 186;
                    106, 81, 163;
                    74, 20, 134];
            case 8
                cmapOut = [252, 251, 253;
                    239, 237, 245;
                    218, 218, 235;
                    188, 189, 220;
                    158, 154, 200;
                    128, 125, 186;
                    106, 81, 163;
                    74, 20, 134];
            case 9
                cmapOut = [252, 251, 253;
                    239, 237, 245;
                    218, 218, 235;
                    188, 189, 220;
                    158, 154, 200;
                    128, 125, 186;
                    106, 81, 163;
                    84, 39, 143;
                    63, 0, 125];
        end
    case 'reds'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [254, 224, 210;
                    252, 146, 114;
                    222, 45, 38];
            case 4
                cmapOut = [254, 229, 217;
                    252, 174, 145;
                    251, 106, 74;
                    203, 24, 29];
            case 5
                cmapOut = [254, 229, 217;
                    252, 174, 145;
                    251, 106, 74;
                    222, 45, 38;
                    165, 15, 21];
            case 6
                cmapOut = [254, 229, 217;
                    252, 187, 161;
                    252, 146, 114;
                    251, 106, 74;
                    222, 45, 38;
                    165, 15, 21];
            case 7
                cmapOut = [254, 229, 217;
                    252, 187, 161;
                    252, 146, 114;
                    251, 106, 74;
                    239, 59, 44;
                    203, 24, 29;
                    153, 0, 13];
            case 8
                cmapOut = [255, 245, 240;
                    254, 224, 210;
                    252, 224, 210;
                    252, 146, 114;
                    251, 106, 74;
                    239, 59, 44;
                    203, 24, 29;
                    153, 0, 13];
            case 9
                cmapOut = [255, 245, 240;
                    254, 224, 210;
                    252, 187, 161;
                    252, 146, 114;
                    251, 106, 74;
                    239, 59, 44;
                    203, 24, 29;
                    165, 15, 21;
                    103, 0, 13];
        end
    case 'bluegreen'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [229, 245, 249;
                    153, 216, 201;
                    44, 162, 95];
            case 4
                cmapOut = [237, 248, 251;
                    178, 226, 226;
                    102, 194, 164;
                    35, 139, 69];
            case 5
                cmapOut = [237, 248, 251;
                    178, 226, 226;
                    102, 194, 164;
                    44, 162, 95;
                    0, 109, 44];
            case 6
                cmapOut = [237, 248, 251;
                    204, 236, 230;
                    153, 216, 201;
                    102, 194, 164;
                    44, 162, 95;
                    0, 109, 44];
            case 7
                cmapOut = [237, 248, 251;
                    204, 236, 230;
                    153, 216, 201;
                    102, 194, 164;
                    65, 174, 118;
                    35, 139, 69;
                    0, 88, 36];
            case 8
                cmapOut = [247, 252, 253;
                    229, 245, 249;
                    204, 236, 230;
                    153, 216, 201;
                    102, 194, 164;
                    65, 174, 118;
                    35, 139, 69;
                    0, 88, 36];
            case 9
                cmapOut = [247, 252, 253;
                    229, 245, 249;
                    204, 236, 230;
                    153, 216, 201;
                    102, 194, 164;
                    65, 174, 118;
                    35, 139, 69;
                    0, 109, 44;
                    0, 68, 27];
        end
    case 'bluepurple'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [224, 236, 244;
                    158, 188, 218;
                    136, 86, 167];
            case 4
                cmapOut = [237, 248, 251;
                    179, 205, 227;
                    140, 150, 198;
                    136, 65, 157];
            case 5
                cmapOut = [237, 248, 251;
                    179, 205, 227;
                    140, 150, 198;
                    136, 86, 167;
                    129, 15, 124];
            case 6
                cmapOut = [237, 248, 251;
                    191, 211, 230;
                    158, 188, 218;
                    140, 150, 198;
                    136, 86, 167;
                    129, 15, 124];
            case 7
                cmapOut = [237, 248, 251;
                    191, 211, 230;
                    158, 188, 218;
                    140, 150, 198;
                    140, 107, 177;
                    136, 65, 157;
                    110, 1, 107];
            case 8
                cmapOut = [247, 252, 253;
                    224, 236, 244;
                    191, 211, 230;
                    158, 188, 218;
                    140, 150, 198;
                    140, 107, 177;
                    136, 65, 157;
                    110, 1, 107];
            case 9
                cmapOut = [247, 252, 253;
                    224, 236, 244;
                    191, 211, 230;
                    158, 188, 218;
                    140, 150, 198;
                    140, 107, 177;
                    136, 65, 157;
                    129, 15, 124;
                    77, 0, 75];
        end
    case 'greenblue'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [224, 243, 219;
                    168, 221, 181;
                    67, 162, 202];
            case 4
                cmapOut = [240, 249, 232;
                    186, 228, 188;
                    123, 204, 196;
                    43, 140, 190];
            case 5
                cmapOut = [240, 249, 232;
                    186, 228, 188;
                    123, 204, 196;
                    67, 162, 202;
                    8, 104, 172];
            case 6
                cmapOut = [240, 249, 232;
                    204, 235, 197;
                    168, 221, 181;
                    123, 204, 196;
                    67, 162, 202;
                    8, 104, 172];
            case 7
                cmapOut = [240, 249, 232;
                    204, 235, 197;
                    168, 221, 181;
                    123, 204, 196;
                    78, 179, 211;
                    43, 140, 190;
                    8, 88, 158];
            case 8
                cmapOut = [247, 252, 240;
                    224, 243, 219;
                    204, 235, 197;
                    168, 221, 181;
                    123, 204, 196;
                    78, 179, 211;
                    43, 140, 190;
                    8, 88, 158];
            case 9
                cmapOut = [247, 252, 240;
                    224, 243, 219;
                    204, 235, 197;
                    168, 221, 181;
                    123, 204, 196;
                    78, 179, 211;
                    43, 140, 190;
                    8, 104, 172;
                    8, 64, 129];
        end
    case 'orangered'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [254, 232, 200;
                    253, 187, 132;
                    227, 74, 51];
            case 4
                cmapOut = [254, 240, 217;
                    253, 204, 138;
                    252, 141, 89;
                    215, 48, 31];
            case 5
                cmapOut = [254, 240, 217;
                    253, 204, 138;
                    252, 141, 89;
                    227, 74, 51;
                    179, 0, 0];
            case 6
                cmapOut = [254, 240, 217;
                    253, 212, 158;
                    253, 187, 132;
                    252, 141, 89;
                    227, 74, 51;
                    179, 0, 0];
            case 7
                cmapOut = [254, 240, 217;
                    253, 212, 158;
                    253, 187, 132;
                    252, 141, 89;
                    239, 101, 72;
                    215, 48, 31;
                    153, 0, 0];
            case 8
                cmapOut = [255, 247, 236;
                    254, 232, 200;
                    253, 212, 158;
                    253, 187, 132;
                    252, 141, 89;
                    239, 101, 72;
                    215, 48, 31;
                    153, 0, 0];
            case 9
                cmapOut = [255, 247, 236;
                    254, 232, 200;
                    253, 212, 158;
                    253, 187, 132;
                    252, 141, 89;
                    239, 101, 72;
                    215, 48, 31;
                    179, 0, 0;
                    127, 0, 0];
        end
    case 'purpleblue'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [236, 231, 242;
                    166, 189, 219;
                    43, 140, 190];
            case 4
                cmapOut = [241, 238, 246;
                    189, 201, 225;
                    116, 169, 207;
                    5, 112, 176];
            case 5
                cmapOut = [241, 238, 246;
                    189, 201, 225;
                    116, 169, 207;
                    43, 140, 190;
                    4, 90, 141];
            case 6
                cmapOut = [241, 238, 246;
                    208, 209, 230;
                    166, 189, 219;
                    116, 169, 207;
                    43, 140, 190;
                    4, 90, 141];
            case 7
                cmapOut = [241, 238, 246;
                    208, 209, 230;
                    166, 189, 219;
                    116, 169, 207;
                    54, 144, 192;
                    5, 112, 176;
                    3, 78, 123];
            case 8
                cmapOut = [255, 247, 251;
                    236, 231, 242;
                    208, 209, 230;
                    166, 189, 219;
                    116, 169, 207;
                    54, 144, 192;
                    5, 112, 176;
                    3, 78, 123];
            case 9
                cmapOut = [255, 247, 251;
                    236, 231, 242;
                    208, 209, 230;
                    166, 189, 219;
                    116, 169, 207;
                    54, 144, 192;
                    5, 112, 176;
                    4, 90, 141;
                    2, 56, 88];
        end
    case 'purplebluegreen'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [236, 226, 240;
                    166, 189, 219;
                    28, 144, 153];
            case 4
                cmapOut = [246, 239, 247;
                    189, 201, 225;
                    103, 169, 207;
                    2, 129, 138];
            case 5
                cmapOut = [246, 239, 247;
                    189, 201, 225;
                    103, 169, 207;
                    28, 144, 153;
                    1, 108, 89];
            case 6
                cmapOut = [246, 239, 247;
                    208, 209, 230;
                    166, 189, 219;
                    103, 169, 207;
                    28, 144, 153;
                    1, 108, 89];
            case 7
                cmapOut = [246, 239, 247;
                    208, 209, 230;
                    166, 189, 219;
                    103, 169, 207;
                    54, 144, 192;
                    2, 129, 138;
                    1, 100, 80];
            case 8
                cmapOut = [255, 247, 251;
                    236, 226, 240;
                    208, 209, 230;
                    166, 189, 219;
                    103, 169, 207;
                    54, 144, 192;
                    2, 129, 138;
                    1, 100, 80];
            case 9
                cmapOut = [255, 247, 251;
                    236, 226, 240;
                    208, 209, 230;
                    166, 189, 219;
                    103, 169, 207;
                    54, 144, 192;
                    2, 129, 138;
                    1, 108, 89;
                    1, 70, 54];
        end
    case 'purplered'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [231, 225, 239;
                    201, 148, 199;
                    221, 28, 119];
            case 4
                cmapOut = [241, 238, 246;
                    215, 181, 216;
                    223, 101, 176;
                    206, 18, 86];
            case 5
                cmapOut = [241, 238, 246;
                    215, 181, 216;
                    223, 101, 176;
                    221, 28, 119;
                    152, 0, 67];
            case 6
                cmapOut = [241, 238, 246;
                    212, 185, 218;
                    201, 148, 199;
                    223, 101, 176;
                    221, 28, 119;
                    152, 0, 67];
            case 7
                cmapOut = [241, 238, 246;
                    212, 185, 218;
                    201, 148, 199;
                    223, 101, 176;
                    231, 41, 138;
                    206, 18, 86;
                    145, 0, 63];
            case 8
                cmapOut = [247, 244, 249;
                    231, 225, 239;
                    212, 185, 218;
                    201, 148, 199;
                    223, 101, 176;
                    231, 41, 138;
                    206, 18, 86;
                    145, 0, 63];
            case 9
                cmapOut = [247, 244, 249;
                    231, 225, 239;
                    212, 185, 218;
                    201, 148, 199;
                    223, 101, 176;
                    231, 41, 138;
                    206, 18, 86;
                    152, 0, 67;
                    103, 0, 31];
        end
    case 'redpurple'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [253, 224, 221;
                    250, 159, 181;
                    197, 27, 138];
            case 4
                cmapOut = [254, 235, 226;
                    251, 180, 185;
                    247, 104, 161;
                    174, 1, 126];
            case 5
                cmapOut = [254, 235, 226;
                    251, 180, 185;
                    247, 104, 161;
                    197, 27, 138;
                    122, 1, 119];
            case 6
                cmapOut = [254, 235, 226;
                    252, 197, 192;
                    250, 159, 181;
                    247, 104, 161;
                    197, 27, 138;
                    122, 1, 119];
            case 7
                cmapOut = [254, 235, 226;
                    252, 197, 192;
                    250, 159, 181;
                    247, 104, 161;
                    221, 52, 151;
                    174, 1, 126;
                    122, 1, 119];
            case 8
                cmapOut = [255, 247, 243;
                    253, 224, 221;
                    252, 197, 192;
                    250, 159, 181;
                    247, 104, 161;
                    221, 52, 151;
                    174, 1, 126;
                    122, 1, 119];
            case 9
                cmapOut = [255, 247, 243;
                    253, 224, 221;
                    252, 197, 192;
                    250, 159, 181;
                    247, 104, 161;
                    221, 52, 151;
                    174, 1, 126;
                    122, 1, 119;
                    73, 0, 106];
        end
    case 'yellowgreen'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [247, 252, 185;
                    173, 221, 142;
                    49, 163, 84];
            case 4
                cmapOut = [255, 255, 204;
                    194, 230, 153;
                    120, 198, 121;
                    35, 132, 67];
            case 5
                cmapOut = [255, 255, 204;
                    194, 230, 153;
                    120, 198, 121;
                    49, 163, 84;
                    0, 104, 55];
            case 6
                cmapOut = [255, 255, 204;
                    217, 240, 163;
                    173, 221, 142;
                    120, 198, 121;
                    49, 163, 84;
                    0, 104, 55];
            case 7
                cmapOut = [255, 255, 204;
                    217, 240, 163;
                    173, 221, 142;
                    120, 198, 121;
                    65, 171, 93;
                    35, 132, 67;
                    0, 90, 50];
            case 8
                cmapOut = [255, 255, 229;
                    247, 252, 185;
                    217, 240, 163;
                    173, 221, 142;
                    120, 198, 121;
                    65, 171, 93;
                    35, 132, 67;
                    0, 90, 50];
            case 9
                cmapOut = [255, 255, 229;
                    247, 252, 185;
                    217, 240, 163;
                    173, 221, 142;
                    120, 198, 121;
                    65, 171, 93;
                    35, 132, 67;
                    0, 104, 55;
                    0, 69, 41];
        end
    case 'yellowgreenblue'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [237, 248, 177;
                    127, 205, 187;
                    44, 127, 184];
            case 4
                cmapOut = [255, 255, 204;
                    161, 218, 180;
                    65, 182, 196;
                    34, 94, 168];
            case 5
                cmapOut = [255, 255, 204;
                    161, 218, 180;
                    65, 182, 196;
                    44, 127, 184;
                    37, 52, 148];
            case 6
                cmapOut = [255, 255, 204;
                    199, 233, 180;
                    127, 205, 187;
                    65, 182, 196;
                    44, 127, 184;
                    37, 52, 148];
            case 7
                cmapOut = [255, 255, 204;
                    199, 233, 180;
                    127, 205, 187;
                    65, 182, 196;
                    29, 145, 192;
                    34, 94, 168;
                    12, 44, 132];
            case 8
                cmapOut = [255, 255, 217;
                    237, 248, 177;
                    199, 233, 180;
                    127, 205, 187;
                    65, 182, 196;
                    29, 145, 192;
                    34, 94, 168;
                    12, 44, 132];
            case 9
                cmapOut = [255, 255, 217;
                    237, 248, 217;
                    199, 233, 180;
                    127, 205, 187;
                    65, 182, 196;
                    29, 145, 192;
                    34, 94, 168;
                    37, 52, 148;
                    8, 29, 88];
        end
    case 'yelloworangebrown'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [255, 247, 188;
                    254, 196, 79;
                    217, 95, 14];
            case 4
                cmapOut = [255, 255, 212;
                    254, 217, 142;
                    254, 153, 41;
                    204, 76, 2];
            case 5
                cmapOut = [255, 255, 212;
                    254, 217, 142;
                    254, 153, 41;
                    217, 95, 14;
                    153, 52, 4];
            case 6
                cmapOut = [255, 255, 212;
                    254, 227, 145;
                    254, 196, 79;
                    254, 153, 41;
                    217, 95, 14;
                    153, 52, 4];
            case 7
                cmapOut = [255, 255, 212;
                    254, 227, 145;
                    254, 196, 79;
                    254, 153, 41;
                    236, 112, 20;
                    204, 76, 2;
                    140, 45, 4];
            case 8
                cmapOut = [255, 255, 229;
                    255, 247, 188;
                    254, 227, 145;
                    254, 196, 79;
                    254, 153, 41;
                    236, 112, 20;
                    204, 76, 2;
                    140, 45, 4];
            case 9
                cmapOut = [255, 255, 229;
                    255, 247, 188;
                    254, 227, 145;
                    254, 196, 79;
                    254, 153, 41;
                    236, 112, 20;
                    204, 76, 2;
                    153, 52, 4;
                    102, 37, 6];
        end
    case 'yelloworangered'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [255, 237, 160;
                    254, 178, 76;
                    240, 59, 32];
            case 4
                cmapOut = [255, 255, 178;
                    254, 204, 92;
                    253, 141, 60;
                    227, 26, 28];
            case 5
                cmapOut = [255, 255, 178;
                    254, 204, 92;
                    253, 141, 60;
                    240, 59, 32;
                    189, 0, 38];
            case 6
                cmapOut = [255, 255, 178;
                    254, 217, 118;
                    254, 178, 76;
                    253, 141, 60;
                    240, 59, 32;
                    189, 0, 38];
            case 7
                cmapOut = [255, 255, 178;
                    254, 217, 118;
                    254, 178, 76;
                    253, 141, 60;
                    252, 78, 42;
                    227, 26, 28;
                    177, 0, 38];
            case 8
                cmapOut = [255, 255, 204;
                    255, 237, 160;
                    254, 217, 118;
                    254, 178, 76;
                    253, 141, 60;
                    252, 78, 42;
                    227, 26, 28;
                    177, 0, 38];
            case 9
                cmapOut = [255, 255, 204;
                    255, 237, 160;
                    254, 217, 118;
                    254, 178, 76;
                    253, 141, 60;
                    252, 78, 42;
                    227, 26, 28;
                    189, 0, 38;
                    128, 0, 38];
        end
    case 'browngreen'
        if numGrads > 9, numGrads = 9; end
        switch numGrads
            case 3
                cmapOut = [216,179,101;
                            245,245,245;
                            90,180,172];
            case 4
                cmapOut = [166,97,26;
                            223,194,125;
                            128,205,193;
                            1,133,113];
            case 5
                cmapOut = [166,97,26;
                            223,194,125;
                            245,245,245;
                            128,205,193;
                            1,133,113];
            case 6
                cmapOut = [140, 81, 10;
                            216, 179, 101;
                            246, 232, 195;
                            199, 234, 229;
                            90, 180, 172;
                            1, 102, 94];
            case 7
                cmapOut = [140, 81, 10;
                            216, 179, 101;
                            246, 232, 195;
                            245, 245, 245;
                            199, 234, 229;
                            90, 180, 172;
                            1, 102, 94];
            case 8
                cmapOut = [140, 81, 10;
                            191, 129, 45;
                            223, 129, 125;
                            246, 232, 195;
                            199, 234, 229;
                            128, 205, 193;
                            53, 151, 143;
                            1, 102, 94];
            case 9
                cmapOut = [140, 81, 10;
                            191, 129, 45;
                            223, 194, 125;
                            246, 232, 195;
                            245, 245, 245;
                            199, 234, 229;
                            128, 205, 193;
                            53, 151, 143;
                            1, 102, 94];
            case 10
                cmapOut = [84, 48, 5;
                            140, 81, 10;
                            191, 129, 45;
                            223, 129, 125;
                            246, 232, 195;
                            199, 234, 229;
                            128, 205, 193;
                            53, 151, 143;
                            1, 102, 94;
                            0, 60, 48];
            case 11
                cmapOut = [84, 48, 5;
                            140, 81, 10;
                            191, 129, 45;
                            223, 129, 125;
                            246, 232, 195;
                            245, 245, 245;
                            199, 234, 229;
                            128, 205, 193;
                            53, 151, 143;
                            1, 102, 94;
                            0, 60, 48];
        end
    case 'pinkgreen'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [233, 163, 201;
                            247, 247, 247;
                            161, 215, 106];
            case 4
                cmapOut = [208, 28, 139;
                            241, 182, 218;
                            184, 225, 134;
                            77, 172, 38];
            case 5
                cmapOut = [208, 28, 139;
                            241, 182, 218;
                            247, 247, 247;
                            184, 225, 134;
                            77, 172, 38];
            case 6
                cmapOut = [197, 27, 125;
                            233, 163, 201;
                            253, 224, 239;
                            230, 245, 208;
                            161, 215, 106;
                            77, 146, 33];
            case 7
                cmapOut = [197, 27, 125;
                            233, 163, 201;
                            253, 224, 239;
                            247, 247, 247;
                            230, 245, 208;
                            161, 215, 106;
                            77, 146, 33];
            case 8
                cmapOut = [197, 27, 125;
                            222, 119, 174;
                            241, 182, 218;
                            253, 224, 239;
                            230, 245, 208;
                            184, 225, 134;
                            127, 188, 65;
                            77, 146, 33];
            case 9
                cmapOut = [197, 27, 125;
                            222, 119, 174;
                            241, 182, 218;
                            253, 224, 239;
                            247, 247, 247;
                            230, 245, 208;
                            184, 225, 134;
                            127, 188, 65;
                            77, 146, 33];
            case 10
                cmapOut = [142, 1, 82;
                            197, 27, 125;
                            222, 119, 174;
                            241, 182, 218;
                            253, 224, 239;
                            230, 245, 208;
                            184, 225, 134;
                            127, 188, 65;
                            77, 146, 33;
                            39, 100, 25];
            case 11
                cmapOut = [142, 1, 82;
                            197, 27, 125;
                            222, 119, 174;
                            241, 182, 218;
                            253, 224, 239;
                            247, 247, 247;
                            230, 245, 208;
                            184, 225, 134;
                            127, 188, 65;
                            77, 146, 33;
                            39, 100, 25];
        end
    case 'purplegreen'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [175, 141, 195;
                            247, 247, 247;
                            127, 191, 123];
            case 4
                cmapOut = [123, 50, 148;
                            194, 165, 207;
                            166, 219, 160;
                            0, 136, 55];
            case 5
                cmapOut = [123, 50, 148;
                            194, 165, 207;
                            247, 247, 247;
                            166, 219, 160;
                            0, 136, 55];
            case 6
                cmapOut = [118, 42, 131;
                            175, 141, 195;
                            231, 212, 232;
                            217, 240, 211;
                            127, 191, 123;
                            27, 120, 55];
            case 7
                cmapOut = [118, 42, 131;
                            175, 141, 195;
                            231, 212, 232;
                            247, 247, 247;
                            217, 240, 211;
                            127, 191, 123;
                            27, 120, 55];
            case 8
                cmapOut = [118, 42, 131;
                            153, 112, 171;
                            194, 165, 207;
                            231, 212, 232;
                            217, 240, 211;
                            166, 219, 160;
                            90, 174, 97;
                            27, 120, 55];
            case 9
                cmapOut = [118, 42, 131;
                            153, 112, 171;
                            194, 165, 207;
                            231, 212, 232;
                            247, 247, 247;
                            217, 240, 211;
                            166, 219, 160;
                            90, 174, 97;
                            27, 120, 55];
            case 10
                cmapOut = [64, 0, 75;
                            118, 42, 131;
                            153, 112, 171;
                            194, 165, 207;
                            231, 212, 232;
                            217, 240, 211;
                            166, 219, 160;
                            90, 174, 97;
                            27, 120, 55;
                            0, 68, 27];
            case 11
                cmapOut = [64, 0, 75;
                            118, 42, 131;
                            153, 112, 171;
                            194, 165, 207;
                            231, 212, 232;
                            247, 247, 247;
                            217, 240, 211;
                            166, 219, 160;
                            90, 174, 97;
                            27, 120, 55;
                            0, 68, 27];
        end
    case 'purpleorange'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [241, 163, 64;
                            247, 247, 247;
                            153, 142, 195];
            case 4
                cmapOut = [230, 97, 1;
                            253, 184, 99;
                            178, 171, 210;
                            94, 60, 153];
            case 5
                cmapOut = [230, 97, 1;
                            253, 184, 99;
                            247, 247, 247;
                            178, 171, 210;
                            94, 60, 153];
            case 6
                cmapOut = [230, 97, 1;
                            241, 163, 64;
                            254, 224, 182;
                            216, 218, 235;
                            153, 142, 195;
                            84, 39, 136];
            case 7
                cmapOut = [230, 97, 1;
                            241, 163, 64;
                            254, 224, 182;
                            247, 247, 247;
                            216, 218, 235;
                            153, 142, 195;
                            84, 39, 136];
            case 8
                cmapOut = [179, 88, 6;
                            224, 130, 20;
                            253, 184, 99;
                            254, 224, 182;
                            216, 218, 235;
                            178, 171, 210;
                            128, 115, 172;
                            84, 39, 136];
            case 9
                cmapOut = [179, 88, 6;
                            224, 130, 20;
                            253, 184, 99;
                            254, 224, 182;
                            247, 247, 247;
                            216, 218, 235;
                            178, 171, 210;
                            128, 115, 172;
                            84, 39, 136];
            case 10
                cmapOut = [127, 59, 8;
                            179, 88, 6;
                            224, 130, 20;
                            253, 184, 99;
                            254, 224, 182;
                            216, 218, 235;
                            178, 171, 210;
                            128, 115, 172;
                            84, 39, 136;
                            45, 0, 75];
            case 11
                cmapOut = [127, 59, 8;
                            179, 88, 6;
                            224, 130, 20;
                            253, 184, 99;
                            254, 224, 182;
                            247, 247, 247;
                            216, 218, 235;
                            178, 171, 210;
                            128, 115, 172;
                            84, 39, 136;
                            45, 0, 75];
        end
    case 'redblue'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [239, 138, 98;
                        247, 247, 247;
                        103, 169, 207];
            case 4
                cmapOut = [202, 0, 32;
                    244, 165, 130;
                    146, 197, 222;
                    5, 113, 176];
            case 5
                cmapOut = [202, 0, 32;
                    244, 165, 130;
                    247, 247, 247;
                    146, 197, 222;
                    5, 113, 176];
            case 6
                cmapOut = [178, 24, 43;
                    239, 138, 98;
                    253, 219, 199;
                    209, 229, 240;
                    103, 169, 207;
                    33, 102, 172];
            case 7
                cmapOut = [178, 24, 43;
                    239, 138, 98;
                    253, 219, 199;
                    247, 247, 247;
                    209, 229, 240;
                    103, 169, 207;
                    33, 102, 172];
            case 8
                cmapOut = [178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    209, 229, 240;
                    146, 197, 222;
                    67, 147, 195;
                    33, 102, 172];
            case 9
                cmapOut = [178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    247, 247, 247;
                    209, 229, 240;
                    146, 197, 222;
                    67, 147, 195;
                    33, 102, 172];
            case 10
                cmapOut = [103, 0, 31;
                    178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    209, 229, 240;
                    146, 197, 222;
                    67, 147, 195;
                    33, 102, 172;
                    5, 48, 97];
            case 11
                cmapOut = [103, 0, 31;
                    178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    247, 247, 247;
                    209, 229, 240;
                    146, 197, 222;
                    67, 147, 195;
                    33, 102, 172;
                    5, 48, 97];
        end
    case 'redgray'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [239, 138, 98;
                    255, 255, 255;
                    153, 153, 153];
            case 4
                cmapOut = [202, 0, 32;
                    244, 165, 130;
                    186, 186, 186;
                    64, 64, 64];
            case 5
                cmapOut = [202, 0, 32;
                    244, 165, 130;
                    255, 255, 255;
                    186, 186, 186;
                    64, 64, 64];
            case 6
                cmapOut = [178, 24, 43;
                    239, 138, 98;
                    253, 219, 199;
                    224, 224, 224;
                    153, 153, 153;
                    77, 77, 77];
            case 7
                cmapOut = [178, 24, 43;
                    239, 138, 98;
                    253, 219, 199;
                    255, 255, 255;
                    224, 224, 224;
                    153, 153, 153;
                    77, 77, 77];
            case 8
                cmapOut = [178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    224, 224, 224;
                    186, 186, 186;
                    135, 135, 135;
                    77, 77, 77];
            case 9
                cmapOut = [178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    255, 255, 255;
                    224, 224, 224;
                    186, 186, 186;
                    135, 135, 135;
                    77, 77, 77];
            case 10
                cmapOut = [103, 0, 31;
                    178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    224, 224, 224;
                    186, 186, 186;
                    135, 135, 135;
                    77, 77, 77;
                    26, 26, 26];
            case 11
                cmapOut = [103, 0, 31;
                    178, 24, 43;
                    214, 96, 77;
                    244, 165, 130;
                    253, 219, 199;
                    255, 255, 255;
                    224, 224, 224;
                    186, 186, 186;
                    135, 135, 135;
                    77, 77, 77;
                    26, 26, 26];
        end
    case 'redyellowblue'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [252, 141, 89;
                    255, 255, 191;
                    145, 191, 219];
            case 4
                cmapOut = [215, 25, 28;
                    253, 174, 97;
                    171, 217, 233;
                    44, 123, 182];
            case 5
                cmapOut = [215, 25, 28;
                    253, 174, 97;
                    255, 255, 191;
                    171, 217, 233;
                    44, 123, 182];
            case 6
                cmapOut = [215, 48, 39;
                    252, 141, 89;
                    254, 224, 144;
                    224, 243, 248;
                    145, 191, 219;
                    69, 117, 180];
            case 7
                cmapOut = [215, 48, 39;
                    252, 141, 89;
                    254, 224, 144;
                    255, 255, 191;
                    224, 243, 248;
                    145, 191, 219;
                    69, 117, 180];
            case 8
                cmapOut = [215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 144;
                    224, 243, 248;
                    171, 217, 233;
                    116, 173, 209;
                    69, 117, 180];
            case 9
                cmapOut = [215, 48, 39;
                            244, 109, 67;
                            253, 174, 97;
                            254, 224, 144;
                            255, 255, 191;
                            224, 243, 248;
                            171, 217, 233;
                            116, 173, 209;
                            69, 117, 180];
            case 10
                cmapOut = [165, 0, 38;
                    215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 144;
                    224, 243, 248;
                    171, 217, 233;
                    116, 173, 209;
                    69, 117, 180;
                    49, 54, 149];
            case 11
                cmapOut = [165, 0, 38;
                    215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 144;
                    255, 255, 191;
                    224, 243, 248;
                    171, 217, 233;
                    116, 173, 209;
                    69, 117, 180;
                    49, 54, 149];
        end
    case 'redyellowgreen'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [252, 141, 89;
                    255, 255, 191;
                    145, 207, 96];
            case 4
                cmapOut = [215, 25, 28;
                    253, 174, 97;
                    166, 217, 106;
                    26, 150, 65];
            case 5
                cmapOut = [215, 25, 28;
                    253, 174, 97;
                    255, 255, 191;
                    166, 217, 106;
                    26, 150, 65];
            case 6
                cmapOut = [215, 48, 39;
                    252, 141, 89;
                    254, 224, 139;
                    217, 239, 139;
                    145, 207, 96;
                    26, 152, 80];
            case 7
                cmapOut = [215, 48, 39;
                    252, 141, 89;
                    254, 224, 139;
                    255, 255, 191;
                    217, 239, 139;
                    145, 207, 96;
                    26, 152, 80];
            case 8
                cmapOut = [215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    217, 239, 139;
                    166, 217, 106;
                    102, 189, 99;
                    26, 152, 80];
            case 9
                cmapOut = [215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    255, 255, 191;
                    217, 239, 139;
                    166, 217, 106;
                    102, 189, 99;
                    26, 152, 80];
            case 10
                cmapOut = [165, 0, 38;
                    215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    217, 239, 139;
                    166, 217, 106;
                    102, 189, 99;
                    26, 152, 80;
                    0, 104, 55];
            case 11
                cmapOut = [165, 0, 38;
                    215, 48, 39;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    255, 255, 191;
                    217, 239, 139;
                    166, 217, 106;
                    102, 189, 99;
                    26, 152, 80;
                    0, 104, 55];
        end
    case 'spectral'
        if numGrads > 11, numGrads = 11; end
        switch numGrads
            case 3
                cmapOut = [252, 141, 89;
                    255, 255, 191;
                    153, 213, 148];
            case 4
                cmapOut = [215, 25, 28;
                    253, 174, 97;
                    171, 221, 164;
                    43, 131, 186];
            case 5
                cmapOut = [215, 25, 28;
                    253, 174, 97;
                    255, 255, 191;
                    171, 221, 164;
                    43, 131, 186];
            case 6
                cmapOut = [213, 62, 79;
                    252, 141, 89;
                    254, 224, 139;
                    230, 245, 152;
                    153, 213, 148;
                    50, 136, 189];
            case 7
                cmapOut = [213, 62, 79;
                    252, 141, 89;
                    254, 224, 139;
                    255, 255, 191;
                    230, 245, 152;
                    153, 213, 148;
                    50, 136, 189];
            case 8
                cmapOut = [213, 62, 79;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    230, 245, 152;
                    171, 221, 164;
                    102, 194, 165;
                    50, 136, 189];
            case 9
                cmapOut = [213, 62, 79;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    255, 255, 191;
                    230, 245, 152;
                    171, 221, 164;
                    102, 194, 165;
                    50, 136, 189];
            case 10
                cmapOut = [158, 1, 66;
                    213, 62, 79;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    230, 245, 152;
                    171, 221, 164;
                    102, 194, 165;
                    50, 136, 189;
                    94, 79, 162];
            case 11
                cmapOut = [158, 1, 66;
                    213, 62, 79;
                    244, 109, 67;
                    253, 174, 97;
                    254, 224, 139;
                    255, 255, 191;
                    230, 245, 152;
                    171, 221, 164;
                    102, 194, 165;
                    50, 136, 189;
                    94, 79, 162];
        end
    case 'accent'
        if numGrads > 8, numGrads = 8; end
        cmapOut = [127, 201, 127;
            190, 174, 212;
            253, 192, 134;
            255, 255, 153;
            56, 108, 176;
            240, 2, 127;
            191, 91, 23;
            102, 102, 102]; % 8 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'dark2'
        if numGrads > 8, numGrads = 8; end
        cmapOut = [27, 158, 119;
            217, 95, 2;
            117, 112, 179;
            231, 41, 138;
            102, 166, 30;
            230, 171, 2;
            166, 118, 29;
            102, 102, 102]; % 8 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'paired'
        if numGrads > 11, numGrads = 11; end
        cmapOut = [166, 206, 227;
            31, 120, 180;
            178, 223, 138;
            51, 160, 44;
            251, 154, 153;
            227, 26, 28;
            253, 191, 111;
            255, 127, 0;
            202, 178, 214;
            106, 61, 154;
            255, 255, 153]; % 11 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'pastel1'
        if numGrads > 9, numGrads = 9; end
        cmapOut = [251, 180, 174;
            179, 205, 227;
            204, 235, 197;
            222, 203, 228;
            254, 217, 166;
            255, 255, 204;
            229, 216, 189;
            253, 218, 236;
            242, 242, 242]; % 9 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'pastel2'
        if numGrads > 8, numGrads = 8; end
        cmapOut = [179, 226, 205;
            253, 205, 172;
            203, 213, 232;
            244, 202, 228;
            230, 245, 201;
            255, 242, 174;
            241, 226, 204;
            204, 204, 204]; % 8 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'set1'
        if numGrads > 9, numGrads = 9; end
        cmapOut = [228, 26, 28;
            55, 126, 184;
            77, 175, 74;
            152, 78, 163;
            255, 127, 0;
            255, 255, 51;
            166, 86, 40;
            247, 129, 191;
            153, 153, 153]; % 9 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'set2'
        if numGrads > 8, numGrads = 8; end
        cmapOut = [102, 194, 165;
            252, 141, 98;
            141, 160, 203;
            231, 138, 195;
            166, 216, 84;
            255, 217, 47;
            229, 196, 148;
            179, 179, 179]; % 8 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'set3'
        if numGrads > 12, numGrads = 12; end
        cmapOut = [141, 211, 199;
            255, 255, 179;
            190, 186, 218;
            251, 128, 114;
            128, 177, 211;
            253, 180, 98;
            179, 222, 105;
            252, 205, 229;
            217, 217, 217;
            188, 128, 189;
            204, 235, 197;
            255, 237, 111]; % 12 class
        cmapOut = cmapOut(1:numGrads,:);
    case 'ben'
        if numGrads > 10, numGrads = 10; end
        cmapOut = 255*[1, 0.4, 0.4; % red
                        0.4, 0.4, 1; % blue
                        [1, 0.7, 0.4]*0.95; % a bit darker orange
                        0.1, 0.5, 0.5; % green
                        [0.9, 0.7, 1]*0.95; % a bit darker pink
                        0.5, 0.1, 0.1;
                        0.5, 0.5, 0.5;
                        0, 0.7, 0;
                        0, 0, 0;
                        1, 1, 0];
        cmapOut = cmapOut(1:numGrads,:);
    otherwise
        error('Unknown color map specified: ''%s''',whichMap);
end

%-------------------------------------------------------------------------------
if flipMe
    cmapOut = flipud(cmapOut/255);
else
    cmapOut = (cmapOut/255);
end

% Convert to the number of colors specified if less than the minimum (3)
if numGrads0 < 3
    cmapOut = cmapOut(1:numGrads0,:);
end

if cellOut
    cmapOut = mat2cell(cmapOut,ones(size(cmapOut,1),1));
end

end