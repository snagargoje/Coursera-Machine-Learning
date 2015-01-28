function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% centroid - Kxm
% X - nxm

n=size(X,1);
m=size(X,2);

% sizeofX=size(X)
% sizeofcen=size(centroids)
% 
% temp = X*centroids'; %nxK
% size(temp)
% temp
% [temp2, idx] = min(temp,[],2);
% 
% idx

for i=1:n
    
    xx = ones(K,1)*X(i,:);
    sizeofxx=size(xx);
    temp = sum( (xx - centroids).*(xx - centroids) , 2 ); 
    
    [val,ind] = min(temp);
    idx(i)=ind;



% =============================================================

end
