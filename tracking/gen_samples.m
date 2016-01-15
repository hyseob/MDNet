function [ bb_samples ] = gen_samples(type, bb, n, opts, trans_f, scale_f)
% GEN_SAMPLES
% Generate sample bounding boxes.
%
% TYPE: sampling method
%   'gaussian'          generate samples from a Gaussian distribution centered at bb
%                       -> positive samples, target candidates                        
%   'uniform'           generate samples from a uniform distribution around bb
%                       -> negative samples
%   'uniform_aspect'    generate samples from a uniform distribution around bb with varying aspect ratios
%                       -> training samples for bbox regression
%   'whole'             generate samples from the whole image
%                       -> negative samples at the initial frame
%
% Hyeonseob Nam, 2015
% 

h = opts.imgSize(1); w = opts.imgSize(2);

% [center_x center_y width height]
sample = [bb(1)+bb(3)/2 bb(2)+bb(4)/2, bb(3:4)];
samples = repmat(sample, [n, 1]);

switch (type)
    case 'gaussian'
        samples(:,1:2) = samples(:,1:2) + trans_f * round(mean(bb(3:4))) * max(-1,min(1,0.5*randn(n,2)));
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*max(-1,min(1,0.5*randn(n,1)))),1,2);
    case 'uniform'
        samples(:,1:2) = samples(:,1:2) + trans_f * round(mean(bb(3:4))) * (rand(n,2)*2-1);
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*(rand(n,1)*2-1)),1,2);
    case 'uniform_aspect'
        samples(:,1:2) = samples(:,1:2) + trans_f * repmat(bb(3:4),n,1) .* (rand(n,2)*2-1);
        samples(:,3:4) = samples(:,3:4) .* opts.scale_factor.^(rand(n,2)*4-2);
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*rand(n,1)),1,2);
    case 'whole'
        range = round([bb(3)/2 bb(4)/2 w-bb(3)/2 h-bb(4)/2]);
        stride = round([bb(3)/5 bb(4)/5]);
        [dx, dy, ds] = meshgrid(range(1):stride(1):range(3), range(2):stride(2):range(4), -5:5);
        windows = [dx(:) dy(:) bb(3)*opts.scale_factor.^ds(:) bb(4)*opts.scale_factor.^ds(:)];
        
        samples = [];
        while(size(samples,1)<n)
            samples = cat(1,samples,...
                windows(randsample(size(windows,1),min(size(windows,1),n-size(samples,1))),:));
        end
end

samples(:,3) = max(10,min(w-10,samples(:,3)));
samples(:,4) = max(10,min(h-10,samples(:,4)));

% [left top width height]
bb_samples = [samples(:,1)-samples(:,3)/2 samples(:,2)-samples(:,4)/2 samples(:,3:4)];
bb_samples(:,1) = max(1-bb_samples(:,3)/2,min(w-bb_samples(:,3)/2, bb_samples(:,1)));
bb_samples(:,2) = max(1-bb_samples(:,4)/2,min(h-bb_samples(:,4)/2, bb_samples(:,2)));
bb_samples = round(bb_samples);


end
