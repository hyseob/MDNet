function [ roidb ] = seq2roidb(config, opts)
% SEQ2ROIDB
% Extract training bounding boxes from the sequence given by config, 
% to construct a dataset of RoIs for training MDNet.
%
% Hyeonseob Nam, 2015
% 

images = config.imgList;
gts = config.gt;

im = imread(images{1});
[h,w,~] = size(im);
imgSize = [h, w];

roidb = sample_rois(images, gts, imgSize, opts);



%--------------------------------------------------------------------------
function rois = sample_rois(images, gts, imgSize, opts)
%--------------------------------------------------------------------------
rois = struct('img_path',cell(1,length(images)),...
    'pos_boxes',cell(1,length(images)),...
    'neg_boxes',cell(1,length(images)));

for i=1:length(images)
    targetLoc = gts(i,:);
%     fprintf('sampling %s ...\n', images{idx});
    
    pos_examples = [];
    while(size(pos_examples,1)<opts.posPerFrame-1)
        pos = genSamples(targetLoc, opts.posPerFrame*5,...
            imgSize, opts.scale_factor, 0.1, 5, false);
        r = overlap_ratio(pos,targetLoc);
        pos = pos(r>opts.posRange(1) & r<=opts.posRange(2),:);
        if isempty(pos), continue; end
        pos = pos(randsample(end,min(end,opts.posPerFrame-1-size(pos_examples,1))),:);
        pos_examples = [pos_examples;pos];
    end
    
    neg_examples = [];
    while(size(neg_examples,1)<opts.negPerFrame)
        neg = genSamples(targetLoc, opts.negPerFrame*2,...
            imgSize, opts.scale_factor, 2, 10, true);
        r = overlap_ratio(neg,targetLoc);
        neg = neg(r>=opts.negRange(1) & r<opts.negRange(2),:);
        if isempty(neg), continue; end
        neg = neg(randsample(end,min(end,opts.negPerFrame-size(neg_examples,1))),:);
        neg_examples = [neg_examples;neg];
    end
    
    rois(i).img_path = images{i};
    rois(i).pos_boxes = single([targetLoc; pos_examples]);
    rois(i).neg_boxes = single(neg_examples);
end



%--------------------------------------------------------------------------
function samples = genSamples(bb, n, imgSize, scale_factor, trans_range, scale_range, valid)
%--------------------------------------------------------------------------
h = imgSize(1); w = imgSize(2);

% [center_x center_y width height]
sample = [bb(1)+bb(3)/2 bb(2)+bb(4)/2, bb(3:4)];
samples = repmat(sample, [n, 1]);

samples(:,1:2) = samples(:,1:2) + trans_range*[bb(3)*(rand(n,1)*2-1) bb(4)*(rand(n,1)*2-1)];
samples(:,3:4) = samples(:,3:4) .* repmat(scale_factor.^(scale_range*(rand(n,1)*2-1)),1,2);
samples(:,3) = max(5,min(w-5,samples(:,3)));
samples(:,4) = max(5,min(h-5,samples(:,4)));

% [left top width height]
samples = [samples(:,1)-samples(:,3)/2 samples(:,2)-samples(:,4)/2 samples(:,3:4)];
if(valid)
    samples(:,1) = max(1,min(w-samples(:,3), samples(:,1)));
    samples(:,2) = max(1,min(h-samples(:,4), samples(:,2)));
else
    samples(:,1) = max(1-samples(:,3)/2,min(w-samples(:,3)/2, samples(:,1)));
    samples(:,2) = max(1-samples(:,4)/2,min(h-samples(:,4)/2, samples(:,2)));
end
samples = round(samples);

        