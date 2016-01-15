function [ imgList ] = parseImg( loc )
% PARSEIMG
% parse image paths from given location
%
% Hyeonseob Nam, 2015
% 

% image extension :
ext = {'jpg', 'png'};

% parse image
tmpList = {};
for i=1:length(ext)
   extList = dir(fullfile(loc, ['*', ext{i}]));
   tmpList = {tmpList{:}, extList(:).name};
end

% put prefix path to imgList
for i=1:length(tmpList)
   tmpList{i} = fullfile(loc, tmpList{i});
end

imgList = tmpList;

end

