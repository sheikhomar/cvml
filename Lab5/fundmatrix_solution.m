% FUNDMATRIX - computes fundamental matrix from 8 or more points
%
% Function computes the fundamental matrix from 8 or more matching points in
% a stereo pair of images.  The normalised 8 point algorithm given by
% Hartley and Zisserman p265 is used.  To achieve accurate results it is
% recommended that 12 or more points are used
%
% Usage:   [F, e1, e2] = fundmatrix(x1, x2)
%          [F, e1, e2] = fundmatrix(x)
%
% Arguments:
%          x1, x2 - Two sets of corresponding 3xN set of homogeneous
%          points.
%         
%          x      - If a single argument is supplied it is assumed that it
%                   is in the form x = [x1; x2]
% Returns:
%          F      - The 3x3 fundamental matrix such that x2'*F*x1 = 0.
%          e1     - The epipole in image 1 such that F*e1 = 0
%          e2     - The epipole in image 2 such that F'*e2 = 0
%

% Copyright (c) 2002-2005 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% Feb 2002  - Original version.
% May 2003  - Tidied up and numerically improved.
% Feb 2004  - Single argument allowed to enable use with RANSAC.
% Mar 2005  - Epipole calculation added, 'economy' SVD used.
% Aug 2005  - Octave compatibility

function [F,e1,e2] = fundmatrix_solution(varargin)
    [x1, x2, npts] = checkargs(varargin(:));

    % Exercise: estimate the fundamental matrix, F, using point correspondences
    % from x1 and x2. Use the principle in slide 8 of
    % 'Lecture4.1-2_StereoEpipolarAndImageAlignment.pdf'. x1 are points in the left
    % image, and x2 are corresponding points in the right image.
    
    nPoints = size(x1,2);
    A = zeros(nPoints,9);
    for n=1:nPoints
        A(n,:) = [x2(1,n)*x1(1,n) x2(1,n)*x1(2,n) x2(1,n) x2(2,n)*x1(1,n) x2(2,n)*x1(2,n) x2(2,n) x1(1,n) x1(2,n) 1];
    end

    [E,V]=eig(A'*A);
    [~,iMin] = min(diag(V));
    F=reshape(E(:,iMin),3,3)';
    
    % Enforce constraint that fundamental matrix has rank 2 by performing
    % a svd and then reconstructing with the two largest singular values.
    [U,D,V] = svd(F,0);
    F = U*diag([D(1,1) D(2,2) 0])*V';

%--------------------------------------------------------------------------
% Function to check argument values and set defaults

function [x1, x2, npts] = checkargs(arg);
    
    if length(arg) == 2
        x1 = arg{1};
        x2 = arg{2};
        if ~all(size(x1)==size(x2))
            error('x1 and x2 must have the same size');
        elseif size(x1,1) ~= 3
            error('x1 and x2 must be 3xN');
        end
        
    elseif length(arg) == 1
        if size(arg{1},1) ~= 6
            error('Single argument x must be 6xN');
        else
            x1 = arg{1}(1:3,:);
            x2 = arg{1}(4:6,:);
        end
    else
        error('Wrong number of arguments supplied');
    end
      
    npts = size(x1,2);
    if npts < 8
        error('At least 8 points are needed to compute the fundamental matrix');
    end
    
