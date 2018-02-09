clear all; close all;

VLFEATROOT = '/home/omar/Downloads/vlfeat-0.9.20';

addpath('MatlabSourceFiles');
run([VLFEATROOT,'/toolbox/vl_setup.m']);

% Load image
I = imread('RSA.jpg');

% Construct scale space (Gaussian pyramid)
[frames, descrp,info] = vl_covdet(single(I),'OctaveResolution',3);
gss = info.gss;
vl_plotss(gss);

octaves = gss.lastOctave-gss.firstOctave+1;
gssSubdivs = gss.octaveLastSubdivision-gss.octaveFirstSubdivision+1;

% Show Difference of Gaussians (DoGs)
figure
DoGs = info.css;
vl_plotss(DoGs);

% Find local extrema
DoGsSubdivs = DoGs.octaveLastSubdivision-DoGs.octaveFirstSubdivision+1;
Maxima = {};
Minima = {};
MaximaNonEdges = {};
MinimaNonEdges = {};
HarrisThreshold = 0.03;
alpha = 0.04;

for o=1:octaves
    [H,W,~] = size(DoGs.data{o});
    for s=2:DoGsSubdivs-1
        fprintf('Calculating local max/min of octave %d/%d, subdivision %d/%d\n',o,octaves,s-1,DoGsSubdivs-2)
        data = DoGs.data{o}(:,:,s-1:s+1);
        localMax = imregionalmax(data);
        localMax = localMax(:,:,2); % Remove lower and upper subdivisions
        localMin = imregionalmin(data);
        localMin = localMin(:,:,2); % Remove lower and upper subdivisions
        
        dataMid = data(:,:,2);
        % Discard local maxima less than 10% of the global maximum
        Ithr = max([0 0.15*max(dataMid(localMax))]);
        localMax(dataMid < Ithr) = 0;
        % Discard local minima larger than 20% of the global minimum
        Ithr = min([0 0.15*min(dataMid(localMin))]);
        localMin(dataMid > Ithr) = 0;
        
        Maxima{o,s-1} = localMax;
        Minima{o,s-1} = localMin;
        
        % Remove edge responses
        [Ix,Iy] = gradient(DoGs.data{o}(:,:,s));
        Ixx = imfilter(Ix.^2,fspecial('gaussian',3,0.5));
        Iyy = imfilter(Iy.^2,fspecial('gaussian',3,0.5));
        Ixy = imfilter(Ix.*Iy,fspecial('gaussian',3,0.5));
        
        % Exercise: modify localMax and localMin by removing blobs whose
        % Harris measures are below 'HarrisThreshold'
        
        % Hint: get a list all found blobs: find(localMax == 1)
        % Hint: remove blob with command: localMax(index) = 0
        
        for x = 1:size(Ix,1)
            for y = 1:size(Ix,2)
                M = [Ixx(x, y) Ixy(x, y); Ixy(x,y) Iyy(x, y)];
                response =  det(M) - alpha * (trace(M)^2);
                if (response < HarrisThreshold)
                    index = sub2ind(size(localMax), x, y);
                    localMax(index) = 0;
                end
            end
        end
        
        
        
        % The modified localMax and localMin are stored for each scale
        MaximaNonEdges{o,s-1} = localMax;
        MinimaNonEdges{o,s-1} = localMin;
    end
end

%% Show a single scale
octave = 2;
suboctave = 1;
figure
subplot(121);imagesc(gss.data{octave}(:,:,suboctave));title('Gaussian');hold on;
showBlobs(Maxima{octave,suboctave},'rx');
showBlobs(Minima{octave,suboctave},'bx');
subplot(122);imagesc(DoGs.data{octave}(:,:,suboctave));title('Difference of Gaussians');hold on;
showBlobs(Maxima{octave,suboctave},'rx');
showBlobs(Minima{octave,suboctave},'bx');

figure
subplot(121);imagesc(DoGs.data{octave}(:,:,suboctave));title('All blobs');hold on;
showBlobs(Maxima{octave,suboctave},'rx');
showBlobs(Minima{octave,suboctave},'bx');
subplot(122);imagesc(DoGs.data{octave}(:,:,suboctave));title('Edges removed');hold on;
showBlobs(MaximaNonEdges{octave,suboctave},'rx');
showBlobs(MinimaNonEdges{octave,suboctave},'bx');