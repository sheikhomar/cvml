clear all;
close all;

% Set up some paths
VLFEATROOT = 'vlfeat-0.9.20';
%run([VLFEATROOT '\toolbox\vl_setup.m']);
addpath([VLFEATROOT,'\toolbox'])
vl_setup()

PTHD        = 'Images';
files       = dir([PTHD '/*.png']);
numframes   = length(files);
r           = ceil(sqrt(numframes)); %for use in debug (visual)

% Load images
for i=1:numframes
    imgArr{i} = imread([PTHD '\hotel.seq' num2str(i-1) '.png']);
    figure(1)
    imshow(imgArr{i});
    drawnow
end

I = double(imgArr{1})/255; % Read first frame

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Harris detector on first frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Harris corner detection on scale 1
R = vl_harris(I,1);

% Find local maxima of R
LocalMax = imregionalmax(R);

% Now suppress local maxima who's corner response is below 30 percent of
% the maximum corner response.
LocalMax(R < 0.3*max(R(:))) = 0;
[y,x]       = ind2sub(size(I),find(LocalMax));
numPoints   = length(x);
Xpts        = x(:);
Ypts        = y(:);

% Display result
figure(2)
imshow(I)
hold on, plot(Xpts(:,1),Ypts(:,1),'r+'), hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do tracking using optical flow (Shi-Tomasi feature tracker)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wsize = 50;
EigMThreshold = 10^-3;

for i=1:numframes-1 % number of images to go through
    disp(['frame ' num2str(i) '/' num2str(numframes)])
    img1 = double(imgArr{i});
    [dx1, dy1] = gradient(double(imgArr{i}));
    img2 = double(imgArr{i+1});
    
    for j=1:numPoints       %number of points to consider
        
        % find window around feature point
        x_center = Xpts(j,i);
        y_center = Ypts(j,i);
        xr = x_center-wsize:x_center+wsize;
        yr = y_center-wsize:y_center+wsize;
        
        [xrange, yrange] = meshgrid(xr,yr);
        
        % Extract patches/windows around the feature point in frame i and
        % i+1
        W1 = interp2(img1, xrange, yrange,'*linear');
        W2 = interp2(img2, xrange, yrange,'*linear');
        W1(isnan(W1))=0;
        W2(isnan(W2))=0;
        
        % Extract gradients from the window around feature point
        Ix = interp2(dx1,xrange,yrange,'*linear');
        Iy = interp2(dy1,xrange,yrange,'*linear');
        Ix(isnan(Ix)) = 0;
        Iy(isnan(Iy)) = 0;
        
        % Do iterative refinement of u and v estimates
        old_xrange  = xrange;
        old_yrange  = yrange;
        oldscore    = sum(sum((W2-W1).^2));
        stopscore   = 9999;
        count = 1;
        xtemp = 0;
        ytemp = 0;
        
        while stopscore > .1 && count < 40;  %iterate until convergence on a point
            % Extract new window around feature point in frame i+1 with new
            % estimates of position.
            Wprime  = interp2(img2, xrange, yrange,'*linear');
            Wprime(isnan(Wprime))=0;
            It      = Wprime-W1; % temporal derivative

            % Construct the matrices 'M' and 'b' to find the displacement (u,v). 
            % Hint: we use 'Ix', 'Iy' and 'It' defined above.
            % We will use these matrices in slide 56 in Lecture12.1-2 Tracking.pdf
            Ixx = sum(Ix(:).^2);
            Iyy = sum(Iy(:).^2);
            Ixy = sum(Ix(:).*Iy(:));

            M = [Ixx Ixy;
                 Ixy Iyy];

            Ixt = sum(Ix(:).*It(:));
            Iyt = sum(Iy(:).*It(:));
            b = -[Ixt;Iyt];      
            
            % Estimate flow
            [~,EigM] = eig(M);
            minEigM = min(diag(EigM));
            
            UVMat   = inv(M)*b;
            UVMat(isnan(UVMat)) = 0;
            u(j,i)  = UVMat(1);
            v(j,i)  = UVMat(2);
            
            % Update window coordinates
            xrange = old_xrange + u(j,i);
            yrange = old_yrange + v(j,i);
            newscore   = sum(sum((W2-Wprime).^2));
            
            % Stop iteration, in case loop goes past optimal (which is not within range of
            %stopping condition)
            if(oldscore < newscore)
                break
            end
            
            stopscore   = abs(oldscore-newscore);
            oldscore    = newscore;
            old_xrange  = xrange;
            old_yrange  = yrange;
            %keep track of path
            
            % Update feature position
            xtemp = xtemp + u(j,i);
            ytemp = ytemp + v(j,i);
            count = count+1;
        end
        
        % Avoid updating the position if M is singular (aperture problem).
        % This corresponds to a small minimum eigenvalue.
        if minEigM >= EigMThreshold
            Xpts(j,i+1) = Xpts(j,i) + xtemp;
            Ypts(j,i+1) = Ypts(j,i) + ytemp;
        else
            Xpts(j,i+1) = Xpts(j,i);
            Ypts(j,i+1) = Ypts(j,i);
        end
        
        % Dissimilarity (fit affine transformation)
        
        % Save the feature window for frame 1.
        if i==1
            Wframe1{j} = W1;
            IxFrame1{j} = Ix;
            IyFrame1{j} = Iy;
        end
        
        % Compute difference between frame i+1 and frame 1 in a window
        % centered at feature point j.
        It = Wprime-Wframe1{j};
        
            % Estimate affine transformation
            A = zeros((2*wsize+1)^2,6);
            b = zeros((2*wsize+1)^2,1);
            for y=-wsize:wsize
                for x=-wsize:wsize
                    IX = IxFrame1{j}(wsize+x+1,wsize+y+1);
                    IY = IyFrame1{j}(wsize+x+1,wsize+y+1);
                    
                    % Exercise: construct a least squares problem that can
                    % estimate the parameters of an affine transformation.
                    % Look at slide 56 for reference.
                    % We want to estimate 'a' in the system of linear
                    % equations: A*a = b
                    % From slide 56, we see that for one pixel in the
                    % window, we have:
                    % Hint: look at the dimensions of 'A' and 'b' defined
                    % above. You should fill out all values in the
                    % matrices.
                    
                    % A(pixelindex,:) = [? ? ? ? ? ?]
                    % b(pixelindex) = -It(pixelindex)
                end
            end
            
            % Solve the linear least squares problem: find a1-a6 parameters 
            if ~all(A == 0)
                a = inv(A'*A)*A'*b;
            else
                a = zeros(6,1);
            end
        
        % Define the affine transformation matrix. Add 1's to the x- and y-
        % coordinates to get: x' = x+u(x,y) and y'=y+v(x,y)
        HAffine = [1+a(2) a(3) a(1);
                    a(5) 1+a(6) a(4);
                    0    0    1];
        
        % Construct (x,y)-coordinates with zero mean (centered in the
        % middle of the window).
        xr = -wsize:wsize;
        yr = -wsize:wsize;
        [xrangeA, yrangeA] = meshgrid(xr,yr);
        V = [ xrangeA(:) yrangeA(:) ones(size(xrangeA(:)))]';
        
        % Perform affine transformation
        V = inv(HAffine)*V;
        
        % Translate coordinates to the center position of the window
        V(1,:) = V(1,:)+Xpts(j,i+1);
        V(2,:) = V(2,:)+Ypts(j,i+1);
        
        % Sample/interpolate these coordinates from the current image
        IAffine = interp2(img2,V(1,:)',V(2,:)');
        IAffine = reshape(IAffine,size(xrangeA));
        IAffine(isnan(IAffine))=0;
        
        patchesAffine{j,i+1} = IAffine;
        
        % Store the window estimated with only translation, for comparison.
        patchesTrans{j,i+1} = Wprime;
        
        % For the first frame, store the original patch of the feature.
        if i == 1
            patchesAffine{j,1} = W1;
            patchesTrans{j,1} = W1;
        end
        
        % --- Calculate a similarity score for each feature point ---
        % Here, we use a correlation metric.
        if i>1
            scoreTrans(j,i) = corr2(patchesTrans{j,i},patchesTrans{j,1}); % Similarity
            scoreAffine(j,i) = corr2(patchesAffine{j,i},patchesAffine{j,1}); % Similarity
        else
            % For the first frame, the correlation is 1, by definition.
            scoreTrans(j,i) = 1;
            scoreAffine(j,i) = 1;
        end
        
    end
    
    imshow(imgArr{i+1})
    hold on, plot(Xpts(:,i+1),Ypts(:,i+1),'r+'), hold off
    drawnow
end

%% Visualize dissimilarities

% Pick at random 7 features
testPoints = randsample([1:numPoints],7);

% Visualize patches tracked with only translation
figure('Name','Translation similarity')
for t=1:length(testPoints)
    k = 1;
    for i=[1:numframes-1]
        subplot(length(testPoints),numframes,(t-1)*numframes+k);
        imagesc(patchesTrans{testPoints(t),i});
        title(num2str(scoreTrans(testPoints(t),i)));axis off
        k = k+1;
    end
end

% Visualize patches tracked with an affine transformation
figure('Name','Affine similarity')
for t=1:length(testPoints)
    k = 1;
    for i=[1:numframes-1]
        subplot(length(testPoints),numframes,(t-1)*numframes+k);
        imagesc(patchesAffine{testPoints(t),i});
        title(num2str(scoreAffine(testPoints(t),i)));axis off
        k = k+1;
    end
end

% Show correlation curves. The x-axis denotes time (frames) and the y-axis
% denotes correlation with the first frame. When a feature correlates less
% than a certain threshold (e.g. 0.9) it is no longer suitable for
% tracking. It is too dissimilar.
figure(5);hold on;
colors = ['y','m','c','r','g','b','k'];
for t=1:length(testPoints)
    plot([1:numframes-1],scoreTrans(testPoints(t),:),[colors(t) '--']);
    plot([1:numframes-1],scoreAffine(testPoints(t),:),colors(t));
end
grid on;


%% Visualize tracked feature points
figure(6)
ix = 1:numframes-1;
ix = [ ix flipdim(ix,2) ix flipdim(ix,2) ];
similarityThr = 0.9;
for i = ix
    imshow(imgArr{i});hold on;
    
    % Extract indices of similar features with correlation >= similarityThr
    similar = scoreAffine(:,i) > similarityThr;
    
    % Green points have features similar with frame 1, red points don't.
    scatter(Xpts(similar,i),Ypts(similar,i),10,'g');
    scatter(Xpts(~similar,i),Ypts(~similar,i),10,'r');
    hold off
    drawnow
end