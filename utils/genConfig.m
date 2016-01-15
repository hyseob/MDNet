function [ config ] = genConfig(dataset,seqName)
% GENCONFIG
% Generate a configuration of a sequence
% 
% INPUT:
%   dataset - The name of dataset ('otb','vot2013','vot2014','vot2015')
%   seqName - The name of a sequence in the given dataset
%
% OUTPUT:
%   config - The configuration of the given sequence
%
% Hyeonseob Nam, 2015
% 

config.dataset = dataset;
config.seqName = seqName;

switch(dataset)
    case {'otb'}
        % path to OTB dataset
        benchmarkSeqHome ='./dataset/OTB/';
        
        % img path
        switch(config.seqName)
            case {'Jogging-1', 'Jogging-2'}
                config.imgDir = fullfile(benchmarkSeqHome, 'Jogging', 'img');
            case {'Skating2-1', 'Skating2-2'}
                config.imgDir = fullfile(benchmarkSeqHome, 'Skating2', 'img');
            otherwise
                config.imgDir = fullfile(benchmarkSeqHome, config.seqName, 'img');
        end
        
        if(~exist(config.imgDir,'dir'))
            error('%s does not exist!!',config.imgDir);
        end
        
        % parse img list
        config.imgList = parseImg(config.imgDir);
        switch(config.seqName)
            case 'David'
                config.imgList = config.imgList(300:end);
            case 'Tiger1'
                config.imgList = config.imgList(6:end);
        end
        
        % load gt
        switch(config.seqName)
            case 'Jogging-1'
                gtPath = fullfile(benchmarkSeqHome, 'Jogging', 'groundtruth_rect.1.txt');
            case 'Jogging-2'
                gtPath = fullfile(benchmarkSeqHome, 'Jogging', 'groundtruth_rect.2.txt');
            case 'Skating2-1'
                gtPath = fullfile(benchmarkSeqHome, 'Skating2', 'groundtruth_rect.1.txt');
            case 'Skating2-2'
                gtPath = fullfile(benchmarkSeqHome, 'Skating2', 'groundtruth_rect.2.txt');
            case 'Human4'
                gtPath = fullfile(benchmarkSeqHome, 'Human4', 'groundtruth_rect.2.txt');
            otherwise
                gtPath = fullfile(benchmarkSeqHome, config.seqName, 'groundtruth_rect.txt');
        end
        
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        gt = importdata(gtPath);
        switch(config.seqName)
            case 'Tiger1'
                gt = gt(6:end,:);
            case {'Board','Twinnings'}
                gt = gt(1:end-1,:);
        end
        config.gt = gt;
        
        nFrames = min(length(config.imgList), size(config.gt,1));
        config.imgList = config.imgList(1:nFrames);
        config.gt = config.gt(1:nFrames,:);
        
    case {'vot2013','vot2014','vot2015'}
        % path to VOT dataset
        benchmarkSeqHome = ['./dataset/VOT/', dataset(end-3:end)];
        
        % img path
        config.imgDir = fullfile(benchmarkSeqHome, config.seqName);
        if(~exist(config.imgDir,'dir'))
            error('%s does not exist!!',config.imgDir);
        end
        
        % parse img list
        images = dir(fullfile(config.imgDir,'*.jpg'));
        images = {images.name}';
        images = cellfun(@(x) fullfile(config.imgDir,x), images, 'UniformOutput', false);
        config.imgList = images;
        
        % gt path
        gtPath = fullfile(benchmarkSeqHome, config.seqName, 'groundtruth.txt');
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        % parse gt
        gt = importdata(gtPath);
        if size(gt,2) >= 6
            x = gt(:,1:2:end);
            y = gt(:,2:2:end);
            gt = [min(x,[],2), min(y,[],2), max(x,[],2) - min(x,[],2), max(y,[],2) - min(y,[],2)];
        end
        config.gt = gt;
        
        nFrames = min(length(config.imgList), size(config.gt,1));
        config.imgList = config.imgList(1:nFrames);
        config.gt = config.gt(1:nFrames,:);
        
    case {'new_dataset'}
        % configure new sequence
end
