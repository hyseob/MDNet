function imo = get_batch(images, boxes, varargin)
% GET_BATCH
% Load, preprocess, and pack images for CNN evaluation
%
% Modified from cnn_imagenet_get_batch() in the MatConvNet library.
% Hyeonseob Nam, 2015
%

opts.input_size     = 107;
opts.crop_mode      = 'warp' ;
opts.crop_padding   = 16 ;
opts.numFetchThreads     = 1 ;
opts.prefetch       = false ;
opts = vl_argparse(opts, varargin);

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

im = cell(1, numel(images)) ;
if opts.numFetchThreads > 0
    if prefetch
        vl_imreadjpeg(images, 'numThreads', opts.numFetchThreads, 'prefetch');
        imo = [];
        return ;
    end
    if fetch
        im = vl_imreadjpeg(images,'numThreads', opts.numFetchThreads) ;
    end
end
if ~fetch
    im = images ;
end

num_boxes = size(boxes, 1);
crop_mode = opts.crop_mode;
crop_size = opts.input_size;
crop_padding = opts.crop_padding;
imo = zeros(crop_size, crop_size, 3, num_boxes, 'single');

parfor i = 1:num_boxes
    id = boxes(i,1);
    bbox = boxes(i,2:end);
    
    imt = im{id};
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt) ;
    end
    
    crop = im_crop(imt, bbox, crop_mode, crop_size, crop_padding);
    imo(:,:,:,i) = crop;
end

