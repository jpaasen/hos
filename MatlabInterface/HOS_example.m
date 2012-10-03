%%
% Example by:
% Jon Petter Åsen - 17.09.2012
% jon.p.asen@ntnu.no - jpaasen@ifi.uio.no

%% INIT variables
fc = 2.5e6; % Hz
c0 = 1540; % m/s

srcTimeStamp = 0; % s
srcPulseLength = 0; % 0 = inf (integer number of ocillations)
obsTimeStamp = 20e-5; % s, there is an known issue with large timestamps compared to element-observation travel times (>0.01s for f=2.5e6 and c0=1540)
useGPUSimulator = true; %(GPU or CPU)

% Array
lambda = c0 / fc;
spacing = 0.5; % lambda
n_srcpoints = 64; % azimuth elements
[coordSrc, width] = getSimpleULA(n_srcpoints, lambda, spacing);    
apod = ones(n_srcpoints, 1);

% delays
%delaysFoc = zeros(1, n_srcpoints);
Rfoc = 0.08; % m
theta = 0; % degrees
theta = theta*pi/180;

% - vector version
focusPoint = Rfoc*[sin(theta), 0, cos(theta)];
srcFocusDist = sqrt(sum((repmat(focusPoint,n_srcpoints,1) - coordSrc).^2, 2));
delays = -(srcFocusDist - max(srcFocusDist)) / c0;

% - geometric version (error in steering calculation)
%delaysFoc = ( sqrt((Rfoc.^2 + (width/2).^2)) - sqrt(Rfoc.^2 + linspace(-width/2,width/2,n_srcpoints).^2) ) / c0;
%delaysSteer = ( linspace(0, width, n_srcpoints)*tan(theta) ) / c0;
%delays = delaysSteer + delaysFoc;
figure(3), stem(delays); title('Delays');

% Observation  areas
% lateral line through diffraction focus
%ulc = [-0.03, 0, 0.08]; % upper left corner
%lrc = [ 0.03, 0, 0.08];%[0.03 0 0.12]; % lower right corner
%n_points = [1000, 1];

% axial line through focus
%ulc = [0, 0, 0]; % upper left corner
%lrc = [0, 0, 0.18]; % lower right corner
%n_points = [1, 100];

% azimuth plane
ulc = [-0.0225 0 0.01];%[-0.09 0 0.01];
lrc = [0.0225 0 0.19];%[0.09 0 0.19];
n_obs = [250 1000];%[1000 1000];
N_obs = n_obs(1)*n_obs(2);  

[coordObs X Z] = getObsCoords(ulc, lrc, n_obs); % Important! coordObs = [X(:), Y(:), Z(:)]

% stack ULA's in az and el for directive elements
n_el = 20;
n_az = 5;
h_el = 10; % elevation high in lambda
w_az = spacing*lambda / n_az;
coordSrc = repmat(coordSrc, n_el, 1);
for i = 1:n_el*n_srcpoints
   coordSrc(i,2) = (-floor(n_el/2) + floor((i-1)/n_srcpoints)) * h_el*lambda/n_el; 
end
coordSrc = repmat(coordSrc, n_az, 1);
for i = 1:n_el*n_az*n_srcpoints
   coordSrc(i,1) = coordSrc(i,1) + (-floor(n_az/2) + floor((i-1)/n_srcpoints/n_el)) * w_az; 
end
delays = repmat(delays, n_el*n_az, 1);
apod = repmat(apod, n_el*n_az, 1);

%N = 1; % select only one ula to evaluate
%coordSrc = coordSrc((end-(N*n_srcpoints)+1:end-((N-1)*n_srcpoints)), :);
%delays = delays((end-n_srcpoints+1:end));
%apod = apod((end-n_srcpoints+1:end));

% plot src and obs points
figure(1), plot3(coordSrc(:,1), coordSrc(:,2), coordSrc(:,3), '.b')
%hold on
% uncomment to see observation points
%figure(1), plot3(:,1), coordObs(:,2), coordObs(:,3), '.g')
%hold off
title('Aperture')
xlabel('x [m]'); ylabel('z [m]'); zlabel('y [m]')


%% CALC field energy
field_energy = HOS_run( ...
    coordObs,           ...
    coordSrc,           ... 
    fc,                 ...
    apod,               ...
    delays,             ...
    srcTimeStamp,       ...
    srcPulseLength,     ...
    obsTimeStamp,       ...
    c0,                 ...
    useGPUSimulator);

%% PROCESS field energy
% convert to db
field_energy_abs = abs(field_energy);
field_energy_norm = field_energy_abs ./ max(field_energy_abs);
field_energy_db = 20*log10( field_energy_norm );

field_energy_reshaped = reshape(field_energy_db, n_obs(2), n_obs(1));

%% PLOT field energy
figure(2)
pcolor(X, Z, field_energy_reshaped );
colormap jet
caxis([-40 0])
shading flat %interp
axis image
xlabel('x [m]'); ylabel('z [m]'); zlabel('y [m]')