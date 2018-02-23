
clear all; close all;

VLFEATROOT = 'vlfeat-0.9.20';
%run([VLFEATROOT '\toolbox\vl_setup.m']);
addpath([VLFEATROOT,'\toolbox'])
vl_setup()

CALIBROOT = 'TOOLBOX_calib';
addpath(CALIBROOT);
MATLABFNSROOT = 'MatlabFns';
addpath(genpath(MATLABFNSROOT))

% Define coordinates of the brochure corners in 3D (Origo is the lower left corner)
Coords3D = [0 21 0;
            30 21 0;
            30 0 0;
            0 0 0]';

% Scale all images by a factor (reduces computation time but may compromise
% calibration accuracy)
scaleImages = 0.75;

folder = 'Images';
files = dir([folder '/*.jpg']);
I1 = rgb2gray(imresize(imread([folder '/' files(1).name]),scaleImages));
F = length(files);
p = zeros(F,3,4); % p is defined as: p(F frames, 2D homogenous coordinates, 4 corners)
for f=1:F
    Irgb = imresize(imread([folder '/' files(f).name]),scaleImages);
    I = rgb2gray(Irgb);
    
    % Compute SIFT features
    [features, descriptors] = vl_sift(single(I)) ;
    
    % For the first frame, mark corner points manually
    if f==1
        imshow(Irgb);
        % Save the four coordinate pairs in the matrix, p.
        disp('Mark upper left corner of the brochure using a single mouse click')
        rect=getrect; p(f,:,1) = [rect(1) rect(2) 1 ];
        disp('Mark upper right corner of the brochure using a single mouse click')
        rect=getrect; p(f,:,2) = [rect(1) rect(2) 1 ];
        disp('Mark lower right corner of the brochure using a single mouse click')
        rect=getrect; p(f,:,3) = [rect(1) rect(2) 1 ];
        disp('Mark lower left corner of the brochure using a single mouse click')
        rect=getrect; p(f,:,4) = [rect(1) rect(2) 1 ];
    else % Use SIFT feature matching to find corner points for all remaining images
        
        % Exercise: Find matches between SIFT features in this frame and a
        % previous frame. Estimate a homography based on these matching points,
        % describing the transformation from the previous frame to this
        % frame. Save the resulting transformed coordinates in the variable
        % 'p' according to the definition above.
        % Hint: use the function 'vl_ubcmatch(f_old,f_new)' to find
        % matching features.
        % Hint: use the function 'ransacfithomography(p_old, p_new, 0.1)'
        % to estimate a homography.
        % Hint: save transformed homogenous coordinates into 'p(f,:,:)'.

        % Find matches between SIFT features in this frame and the previous
        % frame
        [matches, scores] = vl_ubcmatch(descriptorsOld,descriptors) ;
        numMatches = size(matches,2) ;
        
        % Find homography that maps coordinates from the previous image to
        % the current image.
        X1 = featuresOld(1:2,matches(1,:)) ;
        X2 = features(1:2,matches(2,:)) ;

        % Fit homography from feature matches (using RANSAC)
        [H, ok] = ransacfithomography(X1, X2, 0.1);
        
        % Transform the previous corner coordinates, p(f-1,:,:), using the
        % estimated homography.
        p_prime = H*squeeze(p(1,:,:));
        % Divide by the homogeneous coordinates
        p_prime = p_prime./repmat(p_prime(3,:),3,1);
        % p_prime = bsxfun(@rdivide,p_prime,p_prime(3,:)); % Fast version
        
        % Save the corner coordinates of the current frame
        p(f,:,:) = p_prime;
        
        % Plot transformed corners
        imshow(cat(2, I1, I)); hold on;
        p1 = squeeze(p(1,:,:)); % corners in frame 1
        p2 = squeeze(p(f,:,:)); % corners in frame f
        p2(1,:) = p2(1,:) + size(I,2) ;
        h = line([p1(1,:) ; p2(1,:)], [p1(2,:) ; p2(2,:)]) ;
        set(h,'linewidth', 1, 'color', 'b','Marker','.','MarkerSize',8,'MarkerEdgeColor','g') ;
        title('Matching features between images and transforming corner coordinates');
        drawnow
    end
    
    % We want to compare all images against the first one. Therefore we
    % save the feature points and their descriptors for frame 1.
    if f==1
        featuresOld = features;
        descriptorsOld = descriptors;
        IOld = I;
    end
end

%% Run camera calibration
% Parameters used by the camera calibration toolbox
no_image = 1; % only provide coordinates for the toolbox
dont_ask = 1; % don't use toolbox GUI
n_ima = F; % number of images

% Manually set the principal point based on the image size
nx = size(I,2);
ny = size(I,1);
cc = [(nx-1)/2;(ny-1)/2];

% Define input feature locations (2D) and corresponding grid locations (3D)
% in the format used by the calibration toolbox.
for f=1:F
    pp = squeeze(p(f,1:2,:));
    eval(['X_' num2str(f) ' = Coords3D;']);
    eval(['x_' num2str(f) ' = pp;']);
end

% Perform calibration
go_calib_optim

% Illustrate extrinsic results.
% Click the button: 'Switch to world-centered view'
ext_calib

%% Draw axes and graphic on top of the brochure
figure(3)

% Define axes in 3D coordinates.
% Column 1: Origo
% Column 2: X-axis
% Column 3: Y-axis
% Column 4: Z-axis
axes3D = [0 10 0  0;
          0 0  10 0;
          0 0  0  10;
          1 1  1  1];

% Define a cube in 3D centered at (0,0,0)
xc = 0;
yc = 0;
zc = 0;
cube3D = [ xc+1 xc+1 xc-1 xc-1 xc+1 xc+1 xc+1 xc-1 xc-1 xc+1 xc+1 xc+1 xc-1 xc-1 xc-1 xc-1;
           yc+1 yc-1 yc-1 yc+1 yc+1 yc+1 yc-1 yc-1 yc+1 yc+1 yc-1 yc-1 yc-1 yc-1 yc+1 yc+1;
           zc-1 zc-1 zc-1 zc-1 zc-1 zc+1 zc+1 zc+1 zc+1 zc+1 zc+1 zc-1 zc-1 zc+1 zc+1 zc-1];
% Make homogeneous coordinates
cube3D = [cube3D;ones(1,size(cube3D,2))];
% Define translation matrix, T, that translates the cube to the center of
% the brochure and on top of it.
T = [eye(3,3),[Coords3D(1,2)/2;Coords3D(2,2)/2;1];[zeros(1,3),1]];
% Perform translation
cube3D = T*cube3D;

for f=1:F
    Irgb = imread([folder '/' files(f).name]);
    Irgb = imresize(Irgb,scaleImages);
    imshow(Irgb);hold on;
    plot(squeeze(p(f,1,:)),squeeze(p(f,2,:)),'rx','MarkerSize',10);
    
    % Get rotation matrix, Rc, and translation vector, Tc, for the current
    % frame
    eval(['Tc = Tc_' num2str(f) ';']);
    eval(['Rc = rodrigues(omc_' num2str(f) ');']);
    
    % Exercise: define a 3D coordinate system using 4 points (origo + 3
    % axes). Transform the points from 3D world coordinates to 2D image
    % coordinates using the estimated camera parameters.
    % Hint: see slide 13-14 in Lecture2_ImageFormationPartI
    % NOTE: The convention of the extrinsic parameters in the toolbox
    % differs from the convention in slide 13-14.
    % The toolbox uses the following convention:
    % P_camera=Rc*P_world+Tc
    
    % Intrinsic camera matrix (also available directly as the variable 'KK'
    % after the calibration procedure. More information on convention:
    % http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
    K = [fc(1)  alpha_c*fc(1)   cc(1);
        0       fc(2)           cc(2);
        0       0               1   ];
    
    % Projection matrix
    P = [1 0 0 0;
        0 1 0 0;
        0 0 1 0];
    
    % Rotation matrix (with homogeneous coordinates)
    R = [Rc,zeros(3,1);
        zeros(1,3) 1];
    
    % Translation matrix (with homogeneous coordinates)
    T = [eye(3,3),Tc;
        zeros(1,3),1];
    
    % Combined transformation. Note the inverse order of R and T compared
    % with slide 13-14 in Lecture2_ImageFormationPartI.
    M = K*P*T*R;
    
    % Perform transformation of axes points
    axes2D = M*axes3D;
    % Divide by homogeneous coordinates
    axes2D = axes2D./repmat(axes2D(3,:),3,1);
    
    % Exercise: Plot the axes as lines on top of the image
    % Hint: plot([x0 x1],[y0 y1])

    plot([axes2D(1,1) axes2D(1,2)],[axes2D(2,1) axes2D(2,2)],'r','LineWidth',2);
    plot([axes2D(1,1) axes2D(1,3)],[axes2D(2,1) axes2D(2,3)],'g','LineWidth',2);
    plot([axes2D(1,1) axes2D(1,4)],[axes2D(2,1) axes2D(2,4)],'b','LineWidth',2);
    
    % Exercise draw a cube on top of the brochure
    % Hint: use the 3D world coordinates defined in the variable 'cube'
    % from Lab 1
    % Note: since we perform a 2D projection, we use a simple plot-function
    % as opposed to the plot3-function in Lab 1.

    cube2D = M*cube3D;
    cube2D = cube2D./repmat(cube2D(3,:),3,1);
    plot(cube2D(1,:),cube2D(2,:),'c-')
    
    pause % press a key to show next frame
end
