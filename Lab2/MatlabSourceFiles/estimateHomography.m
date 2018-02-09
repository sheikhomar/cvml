function [ H ] = estimateHomography( f1,f2,matches )

numMatches = size(matches,2);
X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

% estimate homography (global transformation from I2 to I1)
  %subset = vl_colsubset(1:numMatches, 4) ;
  A = [] ;
  for i = 1:numMatches
    A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
  end
  [U,S,V] = svd(A) ;
  H = reshape(V(:,9),3,3) ;

end

