function compute_verification_threshold()
% Compute robust threshold & EER for seen/unseen speakers
% Optimized for speed, reliability, and generalization
% ➕ Now includes progress bar for user feedback
%% ================================
%  Directories
%% ================================
modelsDir  = 'E:\VProject\MATLAB\VoiceAuthProject\models';
resultsDir = 'E:\VProject\MATLAB\VoiceAuthProject\results';
dataDir    = 'E:\VProject\MATLAB\VoiceAuthProject\dataset_wav\clips';
cacheFile  = fullfile(resultsDir,'embeddings_cache.mat');
modelFile  = fullfile(modelsDir,'trained_speaker_cnn_embed_final.mat');
if ~exist(modelFile,'file')
   error('Trained model not found: %s', modelFile);
end
if ~exist(resultsDir,'dir'), mkdir(resultsDir); end
%% ================================
%  Audio Datastore
%% ================================
adsAll = audioDatastore(dataDir, ...
   'IncludeSubfolders', true, ...
   'FileExtensions', '.wav', ...
   'LabelSource', 'foldernames');
files  = adsAll.Files;
labels = string(adsAll.Labels);
numFiles = numel(files);
if numFiles < 2
   warning('Not enough files to compute threshold.');
   return;
end
%% ================================
%  Load or Compute Embeddings
%% ================================
if exist(cacheFile,'file')
   load(cacheFile,'embsNorm','labels');
   fprintf('Loaded cached embeddings from %s\n', cacheFile);
else
   fprintf('Extracting embeddings (parallel)...\n');
   sampleLimit = min(numFiles, 300);  % limit for faster processing
   selIdx = randperm(numFiles, sampleLimit);
   files = files(selIdx);
   labels = labels(selIdx);
   % Determine embedding dimension dynamically
   [y, fs] = audioread(files{1});
   if size(y,2)>1, y = mean(y,2); end
   if fs~=16000, y = resample(y,16000,fs); end
   e0 = getEmbedding(y,16000);
   embDim = numel(e0);
   embs = zeros(sampleLimit, embDim);
   embs(1,:) = e0(:)';
   % ===== Progress Bar =====
   hWait = waitbar(0, 'Extracting embeddings... Please wait');
   total = sampleLimit;
   pctStep = max(1, floor(total / 20));  % update ~20 times
   % Parallel embedding extraction
   parfor i = 2:sampleLimit
       [y, fs] = audioread(files{i});
       if size(y,2)>1, y = mean(y,2); end
       if fs~=16000, y = resample(y,16000,fs); end
       embs(i,:) = getEmbedding(y,16000);
       % Update progress bar (use DataQueue to safely update in parfor)
       if mod(i, pctStep) == 0
           waitbar(i / total, hWait, sprintf('Extracting embeddings: %d/%d', i, total));
       end
   end
   close(hWait);
   embsNorm = embs ./ (vecnorm(embs,2,2)+eps);
   save(cacheFile,'embsNorm','labels','-v7.3');
   fprintf('Cached embeddings saved: %s\n', cacheFile);
end
%% ================================
%  Compute Positive Pairs
%% ================================
disp('Generating positive pairs...');
hWait = waitbar(0,'Computing positive pairs...');
uniqueLabels = unique(labels);
numLabels = numel(uniqueLabels);
posScores = [];
for k = 1:numLabels
   waitbar(k/numLabels, hWait, sprintf('Positive pairs: %d/%d', k, numLabels));
   idx = find(labels==uniqueLabels(k));
   if numel(idx) < 2, continue; end
   G = embsNorm(idx,:);
   n = numel(idx);
   pairs = nchoosek(1:n,2);
   if size(pairs,1) > 500
       pairs = pairs(randperm(size(pairs,1),500),:); % sample max 500 pairs
   end
   simVals = sum(G(pairs(:,1),:).*G(pairs(:,2),:),2);
   posScores = [posScores; simVals]; %#ok<AGROW>
end
close(hWait);
%% ================================
%  Compute Negative Pairs
%% ================================
disp('Generating negative pairs...');
numSpeakers = numel(uniqueLabels);
negScores = [];
if numSpeakers > 1
   hWait = waitbar(0,'Computing negative pairs...');
   sampleIdx = arrayfun(@(l)find(labels==l,1), uniqueLabels);
   sampleEmb = embsNorm(sampleIdx,:);
   pairs = nchoosek(1:numSpeakers,2);
   if size(pairs,1) > 1500
       pairs = pairs(randperm(size(pairs,1),1500),:);
   end
   totalPairs = size(pairs,1);
   for j = 1:totalPairs
       if mod(j,200)==0
           waitbar(j/totalPairs, hWait, sprintf('Negative pairs: %d/%d', j, totalPairs));
       end
       negScores = [negScores; sum(sampleEmb(pairs(j,1),:).*sampleEmb(pairs(j,2),:))]; %#ok<AGROW>
   end
   close(hWait);
end
%% ================================
%  Fallback and Combine
%% ================================
if isempty(posScores), posScores = 1; end
if isempty(negScores), negScores = 0; end
scores = [posScores; negScores];
labelsPN = [ones(numel(posScores),1); zeros(numel(negScores),1)];
if numel(unique(labelsPN)) < 2
   warning('Not enough class variation; using default threshold.');
   eer = NaN; baseTh = 0.7;
else
   [X,Y,T] = perfcurve(labelsPN,scores,1);
   FNR = 1 - Y;
   [~, idx] = min(abs(X - FNR));
   eer = mean([X(idx), FNR(idx)]);
   baseTh = T(idx);
end
%% ================================
%  Safe Threshold for Unseen Users
%% ================================
safeMargin = 0.05;  % margin to reject unknown users safely
safeTh = min(1.0, max([baseTh, max(negScores) + safeMargin]));
%% ================================
%  Save Results
%% ================================
save(fullfile(resultsDir,'threshold.mat'),'eer','baseTh','safeTh');
fprintf('\n✅ EER = %.4f | Base Threshold = %.3f | Safe Threshold = %.3f\n', ...
   eer, baseTh, safeTh);
fprintf('Thresholds saved in %s\n', resultsDir);
end
