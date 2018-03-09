function [J,h] = steerGaussMultipleAngles(I,theta,sigma,vis)

% STEERGAUSS Implements a steerable Gaussian filter.
%    This m-file can be used to evaluate the first
%    directional derivative of an image, using the
%    method outlined in:
%
%       W. T. Freeman and E. H. Adelson, "The Design
%       and Use of Steerable Filters", IEEE PAMI, 1991.
%
%    [J,H] = STEERGAUSE(I,THETA,SIGMA,VIS) evaluates
%    the directional derivative of the input image I,
%    oriented at THETA degrees with respect to the
%    image rows. The standard deviation of the Gaussian
%    kernel is given by SIGMA (assumed to be equal to
%    unity by default). The filter parameters are 
%    returned to the user in the structure H.
%
%    Note that H is a structure, with the following fields:
%           H.g: 1D Gaussian
%          H.gp: first-derivative of 1D Gaussian
%       H.theta: orientation of filter
%       H.sigma: standard derivation of Gaussian
%
%    Note that the filter support is automatically
%    adjusted (depending on the value of SIGMA).
%
%    In general, the visualization can be enabled 
%    (or disabled) by setting VIS = TRUE (or FALSE).
%    By default, the visualization is disabled.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part I: Assign algorithm parameters.

% Determine necessary filter support (for Gaussian).
Wx = floor((5/2)*sigma);
if Wx < 1
    Wx = 1
end
x = [-Wx:Wx];

% Evaluate 1D Gaussian filter (and its derivative).
g = exp(-(x.^2)/(2*sigma^2));
gp = -(x/sigma).*exp(-(x.^2)/(2*sigma^2));

% Store filter kernels (for subsequent runs).
h.g = g;
h.gp = gp;
h.theta = -theta*(180/pi);
h.sigma = sigma;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part III: Determine oriented filter response.

% Calculate image gradients (using separability).
Ix = conv2(conv2(I,-gp,'same'),g','same');
Iy = conv2(conv2(I,g,'same'),-gp','same');

% Evaluate oriented filter response.
for a = 1:length(theta)
    J(:,:,a) = cos(theta(a))*Ix+sin(theta(a))*Iy;
    if vis
        figure(1); clf; set(gcf,'Name','Oriented Filtering');
        subplot(2,2,1); imagesc(I); axis image; colormap(gray);
        title('Input Image');
        subplot(2,2,2); imagesc(J(:,:,a)); axis image; colormap(gray);
        title(['Filtered Image (\theta = ',num2str(-theta(a)*(180/pi)),'{\circ})']);
        subplot(2,1,2); imagesc(cos(theta(a))*(g'*gp)+sin(theta(a))*(gp'*g));
        axis image; colormap(gray);
        title(['Oriented Filter (\theta = ',num2str(-theta(a)*(180/pi)),'{\circ})']);
        pause
        drawnow
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%