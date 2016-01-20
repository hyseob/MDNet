%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

clear;

conf = genConfig('otb','Diving');
% conf = genConfig('vot2015','ball1');

switch(conf.dataset)
    case 'otb'
        net = fullfile('models','mdnet_vot-otb.mat');
    case 'vot2014'
        net = fullfile('models','mdnet_otb-vot14.mat');
    case 'vot2015'
        net = fullfile('models','mdnet_otb-vot15.mat');
end

result = mdnet_run(conf.imgList, conf.gt(1,:), net);
