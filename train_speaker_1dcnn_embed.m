function train_speaker_1dcnn_embed()
%% train_speaker_1dcnn_embed
% Speaker classification using MFCC features + CNN
% Combines augmentation, balancing, deeper CNN, LR schedule
% MATLAB R2025a compatible
clc; clear; close all;
%% ========== User paths & settings ==========
rootFolder   = "E:\VProject\MATLAB\VoiceAuthProject";
clipsFolder  = fullfile(rootFolder, "dataset_wav","clips");
tsvFile      = fullfile(rootFolder, "dataset","en","validated.tsv");
outDataset   = fullfile(rootFolder, "ODataset");
modelFolder  = fullfile(rootFolder, "models");
checkpointFolder = fullfile(modelFolder,'checkpoints');
if ~exist(outDataset,'dir'), mkdir(outDataset); end
if ~exist(modelFolder,'dir'), mkdir(modelFolder); end
if ~exist(checkpointFolder,'dir'), mkdir(checkpointFolder); end
minFilesPerSpeaker = 2;      % require at least 2 samples per speaker
trainFrac = 0.80;
valFracOfTrain = 0.10;
rng(0);  % reproducible split
%% ========== Step 1: Restructure dataset ==========
fprintf("📂 Restructuring dataset (speaker folders)...\n");
tbl = readtable(tsvFile,'FileType','text','Delimiter','\t');
for i = 1:height(tbl)
  clientID = string(tbl.client_id{i});
  filename = string(tbl.path{i});
  [~, name, ~] = fileparts(filename);
  sourceFile = fullfile(clipsFolder, name + ".wav");
  classFolder = fullfile(outDataset, clientID);
  if ~exist(classFolder,'dir'), mkdir(classFolder); end
  destFile = fullfile(classFolder, name + ".wav");
  if isfile(sourceFile) && ~isfile(destFile)
      copyfile(sourceFile, destFile);
  end
end
disp("✅ Dataset restructuring complete!");
%% ========== Step 2: Create audioDatastore & filter ==========
ads = audioDatastore(outDataset, ...
  'IncludeSubfolders', true, ...
  'FileExtensions', '.wav', ...
  'LabelSource', 'foldernames');
tblCounts = countEachLabel(ads);
validLabels = tblCounts.Label(tblCounts.Count >= minFilesPerSpeaker);
if isempty(validLabels)
  error("No speaker has >= %d files. Reduce minFilesPerSpeaker or check dataset.", minFilesPerSpeaker);
end
mask = ismember(ads.Labels, validLabels);
ads = subset(ads, mask);
tblCounts = countEachLabel(ads);
disp(tblCounts);
%% ========== Step 3: Balance dataset ==========
counts = tblCounts.Count;
targetCount = round(median(counts));
fprintf("⚖️ Balancing classes to ~%d samples each (oversample)\n", targetCount);
files = ads.Files;
labels = ads.Labels;
newFiles = {};
newLabels = {};
for i = 1:height(tblCounts)
  lbl = tblCounts.Label(i);
  idx = find(labels == lbl);
  n = numel(idx);
  files_i = files(idx);
  newFiles = [newFiles; files_i];
  newLabels = [newLabels; repmat(lbl, n, 1)];
  if n < targetCount
      need = targetCount - n;
      pick = randsample(n, need, true);
      newFiles = [newFiles; files_i(pick)];
      newLabels = [newLabels; repmat(lbl, need, 1)];
  end
end
ads = audioDatastore(newFiles, 'Labels', categorical(newLabels));
tblCounts = countEachLabel(ads);
disp(tblCounts);
%% ========== Step 4: Stratified split ==========
numSpeakers = numel(categories(ads.Labels));
if numSpeakers < 2
  error('Need at least 2 speakers for train/test split.');
end
[adsTrainAll, adsTest] = splitEachLabel(ads, trainFrac, 'randomized');
[adsTrain, adsVal] = splitEachLabel(adsTrainAll, 1 - valFracOfTrain, 'randomized');
fprintf("Dataset sizes: Train=%d  Val=%d  Test=%d\n", ...
  numel(adsTrain.Files), numel(adsVal.Files), numel(adsTest.Files));
%% ========== Step 5: Feature extraction ==========
fprintf("🎵 Extracting MFCC features with augmentation for training...\n");
[XTrain, YTrain, maxLen] = extractFeaturesWithAug(adsTrain);
fprintf("🎵 Extracting MFCC features (no aug) for validation...\n");
[XVal, YVal, ~]   = extractFeaturesWithAug(adsVal, maxLen, false);
fprintf("🎵 Extracting MFCC features (no aug) for test...\n");
[XTest, YTest, ~] = extractFeaturesWithAug(adsTest, maxLen, false);
%% ========== Step 6: Normalize features ==========
fprintf("🔄 Normalizing features...\n");
mu = mean(XTrain(:));
sigma = std(XTrain(:), 1);
sigma(sigma==0) = 1;
XTrain = (XTrain - mu) / sigma;
XVal   = (XVal - mu) / sigma;
XTest  = (XTest - mu) / sigma;
%% ========== Step 7: Fix label categories ==========
categoriesList = categories(YTrain);
YTrain = categorical(YTrain, categoriesList);
YVal   = categorical(YVal, categoriesList);
YTest  = categorical(YTest, categoriesList);
numClasses = numel(categoriesList);
%% ========== Step 8: CNN definition ==========
numFeat = size(XTrain,1);
inputSize = [numFeat, maxLen, 1];
layers = [
  imageInputLayer(inputSize,'Name','input','Normalization','none')
  convolution2dLayer([5 5],64,'Padding','same','Name','conv1')
  batchNormalizationLayer('Name','bn1')
  reluLayer('Name','relu1')
  maxPooling2dLayer(2,'Stride',2,'Name','pool1')
  convolution2dLayer(3,128,'Padding','same','Name','conv2')
  batchNormalizationLayer('Name','bn2')
  reluLayer('Name','relu2')
  maxPooling2dLayer(2,'Stride',2,'Name','pool2')
  convolution2dLayer(3,256,'Padding','same','Name','conv3')
  batchNormalizationLayer('Name','bn3')
  reluLayer('Name','relu3')
  maxPooling2dLayer(2,'Stride',2,'Name','pool3')
  dropoutLayer(0.5,'Name','drop1')
  fullyConnectedLayer(512,'Name','fc1')
  reluLayer('Name','relufc1')
  dropoutLayer(0.5,'Name','drop2')
  fullyConnectedLayer(numClasses,'Name','fc_embed')
  softmaxLayer('Name','softmax')
  classificationLayer('Name','classoutput')];
%% ========== Step 9: Training options ==========
miniBatchSize = 32;           % from 2nd code (stable)
maxEpochs = 30;               % from 2nd code (avoid overfitting)
initialLearnRate = 1e-4;      % from 2nd code (stable learning)
options = trainingOptions('adam', ...
  'MaxEpochs', maxEpochs, ...
  'MiniBatchSize', miniBatchSize, ...
  'InitialLearnRate', initialLearnRate, ...
  'LearnRateSchedule','piecewise', ...
  'LearnRateDropFactor',0.5, ...
  'LearnRateDropPeriod',10, ...
  'Shuffle','every-epoch', ...
  'Plots','training-progress', ...
  'Verbose',true, ...
  'ValidationData',{XVal, YVal}, ...
  'ValidationFrequency',floor(numel(YTrain)/miniBatchSize), ...
  'ExecutionEnvironment','auto', ...
  'CheckpointPath', checkpointFolder, ...
  'ValidationPatience',8, ...
  'L2Regularization',1e-4);
%% ========== Step 10: Train ==========
fprintf("🤖 Training network...\n");
net = trainNetwork(XTrain, YTrain, layers, options);
%% ========== Step 11: Evaluate ==========
fprintf("🔍 Evaluating on test set...\n");
YPred = classify(net, XTest);
acc = mean(YPred == YTest);
fprintf("✅ Test Accuracy: %.2f %%\n", acc*100);
figure('Name','Confusion Matrix');
confusionchart(YTest, YPred);
title(sprintf('Test Accuracy = %.2f %%', acc*100));
%% ========== Step 12: Save ==========
modelFile = fullfile(modelFolder, "trained_speaker_cnn_embed_final.mat");
save(modelFile, 'net', 'maxLen', 'mu', 'sigma', 'categoriesList', 'numFeat', 'inputSize');
fprintf("💾 Saved model and preprocessing to %s\n", modelFile);
end
%% ================= Helper Functions =================
function [X, Y, maxLen] = extractFeaturesWithAug(ads, maxLenIn, doAug)
if nargin < 3, doAug = true; end
if nargin < 2, maxLenIn = []; end
reset(ads);
Xraw = {};
Y = {};
while hasdata(ads)
  [audio, info] = read(ads);
  fs = info.SampleRate;
  if size(audio,2) > 1, audio = mean(audio,2); end
  if isempty(audio) || all(audio==0), continue; end
  coeffs = mfcc(audio, fs, 'NumCoeffs', 40, 'LogEnergy', 'Replace');
  feat = coeffs';
  Xraw{end+1} = feat;
  Y{end+1} = char(info.Label);
  if doAug
      augList = augmentAudioSimple(audio, fs);
      for k = 1:numel(augList)
          a = augList{k};
          coeffsA = mfcc(a, fs, 'NumCoeffs', 40, 'LogEnergy', 'Replace');
          Xraw{end+1} = coeffsA';
          Y{end+1} = char(info.Label);
      end
  end
end
allLens = cellfun(@(x) size(x,2), Xraw);
if isempty(maxLenIn)
  maxLen = round(prctile(allLens,95));
else
  maxLen = maxLenIn;
end
fprintf('Padding/truncating to %d frames\n', maxLen);
numFeat = size(Xraw{1},1);
numFiles = numel(Xraw);
X = zeros(numFeat, maxLen, 1, numFiles, 'single');
for i = 1:numFiles
  feat = Xraw{i};
  L = size(feat,2);
  if L < maxLen
      X(:,1:L,1,i) = feat;
  else
      X(:,:,1,i) = feat(:,1:maxLen);
  end
end
Y = categorical(Y);
end
function outList = augmentAudioSimple(audio, fs)
outList = {};
noisePower = var(audio) / (10^(20/10));
noise = sqrt(max(noisePower, eps)) * randn(size(audio));
outList{end+1} = audio + noise;
% speed-up 1.1x
try
  yspeed = resample(audio, 11, 10);
  outList{end+1} = matchLength(yspeed, audio);
end
% slow-down 0.9x
try
  yslow = resample(audio, 9, 10);
  outList{end+1} = matchLength(yslow, audio);
end
if numel(outList) > 2
  outList = outList(1:2);
end
end
function y2 = matchLength(y, yref)
Lref = numel(yref);
Ly = numel(y);
if Ly > Lref
  y2 = y(1:Lref);
elseif Ly < Lref
  y2 = [y; zeros(Lref - Ly, 1)];
else
  y2 = y;
end
end

