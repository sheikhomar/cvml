clear all;
close all;

addpath('MatlabSourceFiles');

% Read test image
I = imread('onion.png');
I = im2double(rgb2gray(I));

% Sigma controls the degree of smoothing
sigma = 2;

% Alpha controls edge thresholding
alfa=0.1;

% Automatically determine necessary filter support (for Gaussian).
Wx = floor((5/2)*sigma);
if Wx < 1
    Wx = 1;
end

% Smooth with a 2D Gaussian filter
h = fspecial('gaussian',[2*Wx+1 2*Wx+1],sigma);
Ismooth = imfilter(I,h,'same');

% Apply Sobel edge filter to find image gradient
sobel_x = [ -1 0 1
            -2 0 2
            -1 0 1 ];
sobel_y = [ -1 -2 -1
             0  0  0
             1  2  1 ];
Ix = imfilter(Ismooth,sobel_x,'same');
Iy = imfilter(Ismooth,sobel_y,'same');

% Norm of the gradient (Combining the X and Y directional derivatives)
Inorm=sqrt(Ix.*Ix+Iy.*Iy);

% Thresholding
I_max=max(Inorm(:));
I_min=min(Inorm(:));
level=alfa*(I_max-I_min)+I_min;
ix = find(Inorm>=level);
mask = zeros(size(Inorm));
mask(ix) = 1;
Ibw=double(mask).*Inorm;

% Display
figure(1)
subplot(2,3,1),imshow(I),title('Input image')
subplot(2,3,2),show_norm_image(Ix),title('Ix')
subplot(2,3,3),show_norm_image(Iy),title('Iy')
subplot(2,3,4),show_norm_image(Inorm);title('Gradient magnitude')
subplot(2,3,5),imshow(mask),title('Initial thresholding')

% Non-maximum suppression (Thin thresholded image using interpolation to
% find the pixels where the norms of gradient are local maximum.)
[n,m]=size(Ibw);
EdgeMap = zeros(n,m);
for i=2:n-1,
    for j=2:m-1,
        if mask(i,j) == 1
            wsize = 3; % window size: used for visualizing the gradients in a neighborhood
            xr = j-wsize:j+wsize;
            yr = i-wsize:i+wsize;
            [xrange, yrange] = meshgrid(xr,yr);
            Z = interp2(Ibw, xrange, yrange,'nearest');
            center = ceil(length(xr)/2);
            
            % Exercise: construct two gradient vectors with length 1
            % for the point (i,j). The x- and y-values should be
            % stored in the vectors, u and v, respectively.
            % Look at slide 30 in Lecture6_EdgesAndLines.pdf.
            % Hint: use the gradient maps, Ix and Iy, and normalize
            % with Inorm.
            
            u = [0 0]; % Change this
            v = [0 0]; % Change this
            
            u = [Ix(i,j)/Inorm(i,j) -Ix(i,j)/Inorm(i,j)];
            v = [Iy(i,j)/Inorm(i,j) -Iy(i,j)/Inorm(i,j)];
            
            if Inorm(i,j) > 0.3*I_max
                figure(2)
                imshow(Z./I_max,'InitialMagnification',10000);
                hold on;
                
                % Exercise: plot the gradient vectors on top of the
                % gradient map.
                % Hint: use quiver(x,y,u,v,1,'r') to plot a red vector
                % positioned at (x,y) with direction (u,v).
                
                quiver([center center],[center center],u,v,1,'r');
                
                % This pauses the execution of the script (for debugging).
                % Comment out to make execution faster...
                disp('Press any key to continue...')
                pause
            end
            
            % Here, we sample/interpolate from the gradient map at the end
            % positions of the estimated vectors.
            ZI=interp2(Z,u+center,v+center);
            
            % Exercise: assign 'EdgeMap(i,j)=1' if the current gradient at 
            % (i,j) is larger than or equal to its two interpolated neighbors.
            % This is the actual non-maximum suppression.

            if Ibw(i,j) >= ZI(1) & Ibw(i,j) >= ZI(2)
                EdgeMap(i,j)=1;
            end
        end
        disp(['i: ',num2str(i),'/',num2str(n),', j: ',num2str(j),'/',num2str(m)])
    end
end

figure(1)
subplot(2,3,6),imshow(EdgeMap),title('After Thinning');
