clear all; close all;

addpath('MatlabSourceFiles');

VLFEATROOT = 'vlfeat-0.9.20';
%run([VLFEATROOT '\toolbox\vl_setup.m']);
addpath([VLFEATROOT,'\toolbox'])
vl_setup()
MATLABFNSROOT = 'MatlabFns';
addpath(genpath(MATLABFNSROOT))

I1 = rgb2gray(imread('left.jpg')); % left image
I2 = rgb2gray(imread('right.jpg')); % right image

% Find SIFT features
[f1, d1] = vl_sift(single(I1)) ;
[f2, d2] = vl_sift(single(I2)) ;
[matches, scores] = vl_ubcmatch(d1, d2) ;

% Create figure and fit its size to screen size
scrsz = get(0,'ScreenSize');
figure('Name', 'Example', 'Position',[1 1 scrsz(3) scrsz(4)])

% Display detected features
subplot(3,4,1),imshow(I1),title('Features on 1st image');
hold on, plot(f1(1,:),f1(2,:),'r.','MarkerSize',3);
subplot(3,4,5),imshow(I2),title('Features on 2nd image');
hold on, plot(f2(1,:),f2(2,:),'r.','MarkerSize',3);

m1 = f1(1:2,matches(1,:));
m2 = f2(1:2,matches(2,:));

% Fit homography from feature matches (using RANSAC)
[H, inliers] = ransacfithomography(m1, m2, 0.01);

% Display putative matches
subplot(3,4,9),imshow(I1),title('Putative matches');
for n = 1:length(m1);
    line([m1(1,n) m2(1,n)], [m1(2,n) m2(2,n)])
end

% Assemble homogeneous feature coordinates for fitting of the
% homography matrix, note that [x,y] corresponds to [col, row]
x1 = [m1(1,:); m1(2,:); ones(1,length(m1))];
x2 = [m2(1,:); m2(2,:); ones(1,length(m1))];

fprintf('Number of inliers was %d (%d%%) \n', ...
    length(inliers),round(100*length(inliers)/length(m1)))
fprintf('Number of putative matches was %d \n', length(m1))

% Display both images overlayed with inlying matched feature points
subplot(3,4, [2 3 4 6 7 8 10 11 12]);
imagesc(double(I1)+double(I2)), title('Inlying matches'), hold on
plot(m1(1,inliers),m1(2,inliers),'r+');
plot(m2(1,inliers),m2(2,inliers),'g+');
line([m1(1,inliers) ;m2(1,inliers)], [m1(2,inliers); m2(2,inliers)],'color',[0 0 1])

% Illustrate how coordinates are transformed using the homography. This
% worked nicely during camera calibration (exercise 2-4). Why does
% it not work here?
fig = vgg_gui_H(I1, I2, H)

%% Find fundamental matrix
% Append ones to make m1 and m2 homogeneous
m1(3,:) = 1;
m2(3,:) = 1;

% Exercise: implement the missing code in 'fundmatrix.m'. The function is
% called repeatedly from the underlying ransac-function.

fittingfn = @fundmatrix_solution; % You need to implement the missing part of this function! (open fundmatrix.m)
distfn    = @funddist;
degenfn   = @(x) 0;

% Normalize features for increasing computational accuracy
[m1, T1] = normalise2dpts(m1);
[m2, T2] = normalise2dpts(m2);

% Parameters for RANSAC
t = 0.001;
s = 8;

% x1 and x2 are 'stacked' to create a 6xN array for ransac
[F, inliers] = ransac([m1; m2], fittingfn, distfn, degenfn, s, t, 0);

% Now do a final least squares fit on the data points considered to
% be inliers.
F = fittingfn(m1(:,inliers), m2(:,inliers));

% Denormalize
F = T2'*F*T1;

% Illustrate the epipolar constraint
fig=vgg_gui_F(I2,I1,F)

%% Rectify the images
[IL, IR, Hleft, Hright, maskL, maskR] = rectifyImages(I1, I2, [f1(1:2, matches(1, inliers))', f2(1:2, matches(2, inliers))'], F);

imgCombined = horzcat(IL,IR);
figure
imshow(imgCombined)

%% Compute dense disparity map
windowSize = 5;
disparities = 128;
disparityMap = disparity(IL,IR,'BlockSize', windowSize,'DisparityRange', [0 disparities], 'Method','SemiGlobal','UniquenessThreshold',30);

disparityMap(disparityMap < 0) = 0;
figure
imagesc(disparityMap);

%% Our own sparse disparity map - implement correspondence search/template matching

% Manually selected feature points in original left image
p = [204 907 905 818 544 534 194;
    501 83  318 247 296 243 248;
    1   1   1   1   1   1   1];
 
% Transform feature points to fit the rectified left image
fL = Hleft*p;
fL = fL./repmat(fL(3,:),3,[]);

% Define error functions:
% Sum of Squared Differences
SSD =@(t1,t2) sum(sum((double(t1)-double(t2)).^2));
% (negative of) Normalized Cross Correlation
NCC =@(t1,t2) -sum(sum((t2-mean(t2(:))).*(t1-mean(t1(:)))))/sqrt(var(t2(:)-mean(t2(:)))*var(t1(:)-mean(t1(:))));

windowSize = 13; % size of patch used for comparing two positions
maxDisparity = 150; % maximum shift in coordinates from left to right image

% Zero-pad the right and left rectified images
ILPad = double(padarray(IL,[floor(windowSize/2),floor(windowSize/2)]));
IRPad = double(padarray(IR,[floor(windowSize/2),floor(windowSize/2)])); 

% Uncomment the following line if you want to to test with SIFT-features 
% instead of manually selected features.
% fL = vl_sift(single(IL),'PeakThresh',15) ;

fR = zeros(2,size(fL,2));
IRPad(IRPad==0) = 255;


for f=1:size(fL,2)
    cF = round(fL(1,f)); % column index for feature
    rF = round(fL(2,f)); % row index for feature
    
    % Exercise: implement a disparity search in the right image 'IR' along
    % the epipolar line corresponding to the feature positioned at [rF,cF] 
    % in the left image 'IL'. 
    % Your task is to run through all possible positions in the
    % right image and calculate a matching error for each. Use the above defined
    % function 'NCC(patch1, patch2)' that takes a patch from each image and
    % calculates an error. Keep the errors in the vector, 'costVec'.
    % Use the zero-padded images 'ILPad' and 'IRPad' along with
    % 'windowSize' to extract patches.
    
    % costVec = [?,?,?,...]
    
    width = size(IR,2);
    costVec = zeros(1,width);
    for c=1:width
        costVec(c) = NCC(double(ILPad(rF:rF+windowSize-1,cF:cF+windowSize-1)),double(IRPad(rF:rF+windowSize-1,c:c+windowSize-1)));
    end
    
    % Use indMin and indMax to extract error responses within a distance of
    % 'maxDisparity' from feature 'f' in 'IL'. 
    % We only look at points in one direction (left)
    % You want to locate the position within this range that has the minimum 
    % error and assign the index to 'matchedInd'.
    indMin = max([1,cF+ceil(windowSize/2)-maxDisparity]);
    indMax = cF+ceil(windowSize/2); 
    
    % Find the index with the minimum cost
    [~,matchedInd] = min(costVec(indMin:indMax));
    matchedInd = matchedInd+indMin-1;
    
    fR(:,f) = [matchedInd;rF];
    
    if 1 % optionally, change this to 0 when you are done debugging
        figure(6);clf
        subplot(211);imshow(cat(2, IL, IR));hold on;
        plot([fL(1,f),fL(1,f)+size(IL,2)],[fL(2,f),fL(2,f)],'r.','MarkerSize',8);
        plot(fR(1,f)+size(IL,2),fR(2,f),'g.','MarkerSize',8);
        h = line([fL(1,f) ; fR(1,f)+size(IL,2)], [fL(2,f) ; fR(2,f)]) ;
        subplot(212);plot(costVec);hold on;
        plot([indMin,indMin],[min(costVec),max(costVec)],'k');
        plot([indMax,indMax],[min(costVec),max(costVec)],'k');
        plot([cF+windowSize/2,cF+windowSize/2],[min(costVec) max(costVec)],'r')
        plot([matchedInd,matchedInd],[min(costVec),max(costVec)],'g');
        legend('error/cost','indMin','indMax','left image','right image')
        pause
    end
    disp([num2str(f) '/' num2str(size(fL,2))]);
end

% Display all matches
figure(7);imshow(cat(2, IL, IR)) ;hold on ;
h = line([fL(1,:) ; fR(1,:) + size(IL,2)], [fL(2,:) ; fR(2,:)]) ;
set(h,'linewidth', 1, 'color', 'b','Marker','.','MarkerSize',8,'MarkerEdgeColor','g') ;


%% Find distances from disparities
f = 0.009; % (made up) focal length
T = 2000; % (made up) baseline

% Exercise: calculate a rough estimate of the distances for each feature.
% Use the correspondences 'fL' and 'fR' in the left and right images,
% respectively. Use the principle in slide 13 of 'Lecture3_StereoEpipolarGeometry.pdf'.
% Remember that the distances are calculated in an arbitrary scale, since
% we do not know the intrinsics of the camera. Therefore, to assess the
% correctness of the found distances, try to look at the relative distances
% of the features.

% Z = [?,?,?,...]

disparities = fL(1,:)-fR(1,:);
Z = f*T./disparities;

% Illustrate estimated depths
figure(8);
Ic = IL;
Ic(:,:,2) = IR;
Ic(:,:,3) = 0;
imshow(Ic);hold on;
h = line([fL(1,:) ; fR(1,:)], [fL(2,:) ; fR(2,:)]) ;
set(h,'linewidth', 1, 'color', 'b') ;
plot(fL(1,:),fL(2,:),'r.','MarkerSize',8);
plot(fR(1,:),fR(2,:),'g.','MarkerSize',8);
text(fL(1,:), fL(2,:), cellstr(num2str(Z')), 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','right', 'FontSize', 10,'Color','w')