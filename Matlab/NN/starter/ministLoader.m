function patches = ministLoader()
% sampleIMAGES
% Returns 10000 patches for training

images = loadMNISTImages('train-images.idx3-ubyte');    % load images from disk 
lables = loadMNISTLabels('train-labels.idx1-ubyte');

numpatches = 10000;


%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = images(:,1:numpatches);

end