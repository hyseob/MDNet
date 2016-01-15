function bbox_reg = train_bbox_regressor(X, bbox, gt, varargin)
% bbox_reg = rcnn_train_bbox_regressor(imdb, rcnn_model, varargin)
%   Trains a bounding box regressor on the image database imdb
%   for use with the R-CNN model rcnn_model. The regressor is trained
%   using ridge regression.
%
%   Keys that can be passed in:
%
%   min_overlap     Proposal boxes with this much overlap or more are used
%   layer           The CNN layer features to regress from (either 5, 6 or 7)
%   lambda          The regularization hyperparameter in ridge regression
%   robust          Throw away examples with loss in the top [robust]-quantile
%   binarize        Binarize features or leave as real values >= 0

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addParamValue('min_overlap', 0.6,   @isscalar);
ip.addParamValue('lambda',      1000,  @isscalar);
ip.addParamValue('robust',      0,     @isscalar);

ip.parse(varargin{:});
opts = ip.Results;

% fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
% fprintf('Training options:\n');
% disp(opts);
% fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% ------------------------------------------------------------------------
% Get all positive examples
[Y, O] = get_examples(bbox, gt);

idx = find(O>opts.min_overlap);
X = X(idx,:); Y = Y(idx,:);

% use ridge regression solved by cholesky factorization
method = 'ridge_reg_chol';

% add bias feature
X = cat(2, X, ones(size(X,1), 1, class(X)));

% Center and decorrelate targets
mu = mean(Y);
Y = bsxfun(@minus, Y, mu);
S = Y'*Y / size(Y,1);
[V, D] = eig(S);
D = diag(D);
T = V*diag(1./sqrt(D+0.001))*V';
T_inv = V*diag(sqrt(D+0.001))*V';
Y = Y * T;

model.mu = mu;
model.T = T;
model.T_inv = T_inv;

model.Beta = [ ...
    solve_robust(X, Y(:,1), opts.lambda, method, opts.robust) ...
    solve_robust(X, Y(:,2), opts.lambda, method, opts.robust) ...
    solve_robust(X, Y(:,3), opts.lambda, method, opts.robust) ...
    solve_robust(X, Y(:,4), opts.lambda, method, opts.robust)];

bbox_reg.model = model;
bbox_reg.training_opts = opts;
% save([conf.cache_dir 'bbox_regressor_final'], 'bbox_reg');


% ------------------------------------------------------------------------
function [Y, O] = get_examples(bbox, gt)
% ------------------------------------------------------------------------
n = size(bbox,1);

% target values
Y = zeros(n, 4, 'single');

% overlap amounts
O = overlap_ratio(bbox,gt);

for i = 1:n
    ex_box = bbox(i,:);
    gt_box = gt(i,:);
    
    src_w = ex_box(3);
    src_h = ex_box(4);
    src_ctr_x = ex_box(1) + 0.5*src_w;
    src_ctr_y = ex_box(2) + 0.5*src_h;
    
    gt_w = gt_box(3);
    gt_h = gt_box(4);
    gt_ctr_x = gt_box(1) + 0.5*gt_w;
    gt_ctr_y = gt_box(2) + 0.5*gt_h;

    dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1/src_w;
    dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1/src_h;
    dst_scl_w = log(gt_w / src_w);
    dst_scl_h = log(gt_h / src_h);

    Y(i, :) = [dst_ctr_x dst_ctr_y dst_scl_w dst_scl_h];
end


% ------------------------------------------------------------------------
function [x, losses] = solve_robust(A, y, lambda, method, qtile)
% ------------------------------------------------------------------------
[x, losses] = solve(A, y, lambda, method);
% fprintf('loss = %.3f\n', mean(losses));
if qtile > 0
  thresh = quantile(losses, 1-qtile);
  I = find(losses < thresh);
  [x, losses] = solve(A(I,:), y(I), lambda, method);
  fprintf('loss (robust) = %.3f\n', mean(losses));
end


% ------------------------------------------------------------------------
function [x, losses] = solve(A, y, lambda, method)
% ------------------------------------------------------------------------

%tic;
switch method
case 'ridge_reg_chol'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  %
  % solve (A'A + lambdaI)x = A'y for x using cholesky factorization
  % R'R = (A'A + lambdaI)
  % R'z = A'y  :  solve for z  =>  R'Rx = R'z  =>  Rx = z
  % Rx = z     :  solve for x
  R = chol(A'*A + lambda*eye(size(A,2)));
  z = R' \ (A'*y);
  x = R \ z;
case 'ridge_reg_inv'
  % solve for x in min_x ||Ax - y||^2 + lambda*||x||^2
  x = inv(A'*A + lambda*eye(size(A,2)))*A'*y;
case 'ls_mldivide'
  % solve for x in min_x ||Ax - y||^2
  if lambda > 0
    warning('ignoring lambda; no regularization used');
  end
  x = A\y;
end
%toc;
losses = 0.5 * (A*x - y).^2;


% ------------------------------------------------------------------------
function r = overlap_ratio(rect1, rect2)
% ------------------------------------------------------------------------
inter_area = diag(rectint(rect1,rect2));
union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;
r = inter_area./union_area;
