function [obs X Z] = getObsCoords(ulc, lrc, no_points)
%GETOBSCOORDS Returns an observation plane.
%   The observation plane is defined by a bounding box [ulc, lrc] and the
%   number of points in each direction.
%
%       ulc = upper left corner
%       lrc = lower right corner
%       no_points in x and z.

ulcX = ulc(1);
ulcZ = ulc(3);

lrcX = lrc(1);
lrcZ = lrc(3);

x = linspace(ulcX,lrcX,no_points(1));
z = linspace(ulcZ,lrcZ,no_points(2));

[X Z] = meshgrid(x, z);
Y = zeros(no_points);

obs = [X(:) Y(:) Z(:)];

end

