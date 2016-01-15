function [net, info] = mdnet_train(net, roidb, getBatch, varargin)
% MDNET_TRAIN
% Train a MDNet by a modified SGD.
%
% Modified from cnn_train() in the MatConvNet library.
% Hyeonseob Nam, 2015
% 

opts.batch_frames = 8 ;
opts.batchSize    = 128 ;
opts.batch_pos    = 32;
opts.batch_neg    = 96;

opts.numCycles    = 100 ;
opts.useGpu       = false ;
opts.conserveMemory = false ;

opts.sync = true ;
opts.learningRate = 0.0001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts = vl_argparse(opts, varargin) ;

K = length(roidb);

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
            class(net.layers{i}.filters)) ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
            class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        
        if ~isfield(net.layers{i}, 'filtersLearningRate')
            net.layers{i}.filtersLearningRate = 1 ;
        end
        if ~isfield(net.layers{i}, 'biasesLearningRate')
            net.layers{i}.biasesLearningRate = 2 ;
        end
        if ~isfield(net.layers{i}, 'filtersWeightDecay')
            net.layers{i}.filtersWeightDecay = 1 ;
        end
        if ~isfield(net.layers{i}, 'biasesWeightDecay')
            net.layers{i}.biasesWeightDecay = 0 ;
        end
        
        if opts.useGpu
            net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum);
            net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum);
        end
        
    end
end

% -------------------------------------------------------------------------
%                                                                  Training
% -------------------------------------------------------------------------
if opts.useGpu
    one = gpuArray(single(1)) ;
    net = vl_simplenn_move(net, 'gpu') ;
else
    one = single(1) ;
    net = vl_simplenn_move(net, 'cpu') ;
end
res = [] ;
lr = opts.learningRate;

% shuffle the frames for training
frame_list = cell(K,1);
for k=1:K
    nFrames = opts.batch_frames*opts.numCycles; 
    while(length(frame_list{k})<nFrames)
        frame_list{k} = cat(2,frame_list{k},uint32(randperm(length(roidb{k}))));
    end
    frame_list{k} = frame_list{k}(1:nFrames);
end

% init info
info.train.objective = zeros(K,opts.numCycles) ;
info.train.error = zeros(K,opts.numCycles) ;
info.train.speed = zeros(K,opts.numCycles) ;

%% training on training set
nextBatch = [];
for t=1:opts.numCycles
    fprintf('Training: processing cycle %3d of %3d ...\n', t, opts.numCycles) ;
    
    for seq_id=1:K
        batch_time = tic ;
        fprintf('\t seq %02d: ',seq_id);
        
        % get next image batch and labels
        if(isempty(nextBatch))
            batch = frame_list{seq_id}((t-1)*opts.batch_frames+1:t*opts.batch_frames);
        else
            batch = nextBatch;
        end
        [im, labels] = getBatch(roidb{seq_id}, batch, opts.batch_pos, opts.batch_neg) ;
        
        if opts.useGpu
            im = gpuArray(im) ;
        end
        
        % backprop
        net.layers{end}.class = labels ;
        res = mdnet_simplenn(net, im, seq_id, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        
        % gradient step
        for l=1:numel(net.layers)
            if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
            
            net.layers{l}.filtersMomentum = ...
                opts.momentum * net.layers{l}.filtersMomentum ...
                - (lr * net.layers{l}.filtersLearningRate) * ...
                (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
                - (lr * net.layers{l}.filtersLearningRate) / opts.batchSize * res(l).dzdw{1} ;
            
            net.layers{l}.biasesMomentum = ...
                opts.momentum * net.layers{l}.biasesMomentum ...
                - (lr * net.layers{l}.biasesLearningRate) * ....
                (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
                - (lr * net.layers{l}.biasesLearningRate) / opts.batchSize * res(l).dzdw{2} ;
            
            net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
            net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
        end
        
        % print information
        batch_time = toc(batch_time) ;
        speed = opts.batchSize/batch_time ;
        info.train = updateError(info.train, t, seq_id, labels, res, batch_time) ;
        
        fprintf(' %.2f s (%.1f images/s),', batch_time, speed) ;
        fprintf(' objective %.3f, error %.3f\n', ...
            info.train.objective(seq_id,t)/(t*opts.batchSize),...
            info.train.error(seq_id,t)/(t*opts.batchSize)) ;
    end
    fprintf('\n') ;
    
    mean_objective = mean(info.train.objective(:,t))/(t*opts.batchSize) ;
    mean_error = mean(info.train.error(:,t))/(t*opts.batchSize) ;
    fprintf('Total: objective %.3f, error %.3f\n', mean_objective, mean_error) ;
    fprintf('\n') ;
end % next batch

info.train.objective = info.train.objective ./ (opts.batchSize*repmat(1:opts.numCycles,K,1)) ;
info.train.error = info.train.error ./ (opts.batchSize*repmat(1:opts.numCycles,K,1))  ;
info.train.speed = (opts.batchSize*repmat(1:opts.numCycles,K,1)) ./ info.train.speed ;



% -------------------------------------------------------------------------
function info = updateError(info, t, k, labels, res, time)
% -------------------------------------------------------------------------

if(t>1)
    info.objective(k,t) = info.objective(k,t-1) + gather(res(end).x) ;
    info.speed(k,t) = info.speed(k,t-1) + time;
else
    info.objective(k,t) = gather(res(end).x) ;
    info.speed(k,t) = time;
end

if(size(res(end-1).x,3)==2)
    predictions = gather(res(end-1).x) ;
else
    predictions = gather(res(end-1).x(:,:,k*2-1:k*2,:)) ;
end
sz = size(predictions) ;
n = prod(sz([1,2])) ;

[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
if(t>1)
    info.error(k,t) = info.error(k,t-1) + sum(sum(sum(error(:,:,1,:))))/n ;
else
    info.error(k,t) = sum(sum(sum(error(:,:,1,:))))/n ;
end


