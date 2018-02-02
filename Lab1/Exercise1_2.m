clear all;  close all;

Miranda_org=imread('miranda.tif');
Miranda_org = imresize(Miranda_org, [400 400]);
imwrite(Miranda_org, 'miranda1.tif');

% create a copy of the image with superimposed slat noise
Miranda_scratch = Miranda_org;
for i= 150:250
	for j=150:250
		if rand>0.9 Miranda_scratch(i,j)=255; end
    end
end

% apply the thresholded median filter (use a varying threshold value)
Miranda_rec = ???

% apply median filtering
Miranda_med = Miranda_scratch;
for i= 150:250
	for j=150:250
		Miranda_med(i,j) = ???
    end
end


figure;
subplot( 2,2,1 );   imshow(Miranda_org);     title('Original image')
subplot( 2,2,2 );   imshow(Miranda_scratch); title('Scratched image')
subplot( 2,2,3 );   imshow(Miranda_rec);     title('Denoised image by median threshold filter')
subplot( 2,2,4 );   imshow(Miranda_med);     title('Denoised image by median filter')
disp(num2str(min(min(double(Miranda_scratch==Miranda_rec)))));