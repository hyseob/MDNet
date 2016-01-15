function r = overlap_ratio(rect1, rect2)
% OVERLAP_RATIO
% Compute the overlap ratio between two rectangles
%
% Hyeonseob Nam, 2015
% 

inter_area = rectint(rect1,rect2);
union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;

r = inter_area./union_area;
end