function [ output_args ] = showBlobs( blobsImage, plotOptions )
[H,W,~] = size(blobsImage);
[Y,X] = ind2sub([H,W],find(blobsImage==1));

plotOps = 'rx';
if nargin > 1
    plotOps = plotOptions;
end

plot(X,Y,plotOps);

end

