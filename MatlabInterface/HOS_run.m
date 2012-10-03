function field_energy = HOS_run(coordObs, coordSrc, fSrc, apodSrc, steerFocusDelaySrc, srcTimeStamp, srcPulseLength, timestampObs, c0, useGPUSimulator)
% Calculate the field energy 
%   Parameters:
%       coordObs,			Observation coordiantes: size = [N 3]
%       coordSrc,			Source coordinates: size = [M 3]
%       fSrc,				Frequency: value > 0, size = 1 
%       apodSrc,			Apodization: size = M
%       steerFocusDelaySrc,	Steer-focus delays: size = M
%       srcTimeStamp,		Time stamp for when source starts to fire: size = 1
%       srcPulseLength,		Pulse length 0 == Inf: size = 1
%       timestampObs,		Current timestamp for this observation: size = 1
%       c0,					Propagation speed: size = 1
%       useGPUSimulator,    Use GPU (true = default) or (CPU) simulator

field_energy = 0;

if ndims(coordObs) ~= 2 || size(coordObs, 2) ~= 3
    disp 'ndims(coordObs) must be 2 and size(coordObs, 1) must be 3'
    return
elseif ndims(coordSrc) ~= 2 || size(coordSrc, 2) ~=3
    disp 'ndims(coordSrc) must be 2 and size(coordSrc, 1) must be 3'
    return
end

%N = size(coordObs, 1);
M = size(coordSrc, 1);

if length(fSrc) ~= 1 || length(timestampObs) ~= 1 || length(c0) ~= 1 || length(srcTimeStamp) ~= 1 || length(srcPulseLength) ~= 1
    disp 'fSrc, timestampObs, srcTimeStamp, srcPulseLength and c0 must be a scalars'
    return
elseif length(apodSrc) ~= M || length(steerFocusDelaySrc) ~= M
    disp 'length of apodSrc and steerFocusDelaySrc must be equal to size(coordSrc,2)'
    return
end

if fSrc <= 0
    disp 'fSrc must be > 0'
    return
end

if nargin < 10
   useGPUSimulator = true; 
end
  
% convert parameters to HOS internal format
fSrc            = fSrc * ones(M,1);
srcTimeStamp    = srcTimeStamp * ones(M,1);
timestampObs    = timestampObs * ones(M,1);
srcPulseLength  = int32(srcPulseLength * ones(M,1));
    
tic

field_energy = HuygensOnSpeed(coordObs, coordSrc, fSrc, apodSrc, steerFocusDelaySrc, srcTimeStamp, srcPulseLength, timestampObs, c0, useGPUSimulator);

toc

end

