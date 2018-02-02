function new=medfilt_th(old,n,thresh);
%
% Input is matrix "old" and the dimensions
% of the nxn median filter mask. The filter
% dimension "n" must be of odd value.The output
% is the new median filtered image.
%
% Dave Johnson (1/22/96) included the
% the parameter thresh is used to define a
% threshold for the filter.  If the difference
% between the median and a data point is less
% than the threshold, then the filter does not
% operate on that point.

[io,jo] = size(old);
new = medfilt2(old,[n n]);
old = double(old);
new = double(new);

% DMJ modification - use threshold to determine whether new
% values are filtered or not
for i=1:io;
  for j=1:jo;
    if abs(old(i,j)-new(i,j)) <= thresh;
      new(i,j)=old(i,j);
    end
  end
end

new = uint8(new);
