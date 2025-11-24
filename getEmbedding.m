function emb = getEmbedding(waveform, fs)
% Return normalized embedding (row vector) from waveform using 'fc_embed' layer
% waveform : column vector (raw audio) OR string/file path OR cell array
% fs       : sampling rate (default = 16kHz)
%
% Loads trained CNN automatically from models folder.

    % === Paths ===
    baseDir   = 'E:\VProject\MATLAB\VoiceAuthProject';
    modelsDir = fullfile(baseDir,'models');
    netFile   = fullfile(modelsDir,'trained_speaker_cnn_embed_final.mat');

    % === Load trained network ===
    persistent net
    if isempty(net)
        S = load(netFile,'net');
        if ~isfield(S,'net')
            error('Trained network not found in %s', netFile);
        end
        net = S.net;
    end

    % === Handle input: file or waveform ===
    if ischar(waveform) || isstring(waveform)
        [y, fsIn] = audioread(waveform);
        if size(y,2) > 1, y = mean(y,2); end
        if nargin < 2, fs = fsIn; end
        waveform = y;
    end

    if nargin < 2
        fs = 16000; % default
    end

    % === Ensure mono + resample ===
    if size(waveform,2) > 1
        waveform = mean(waveform,2);
    end
    if fs ~= 16000
        waveform = resample(waveform,16000,fs);
        fs = 16000;
    end

    % === Wrap in cell if needed ===
    if ~iscell(waveform)
        waveform = {waveform(:)};
    end

    feats = cell(size(waveform));

    % === Extract MFCC features ===
    for i = 1:numel(waveform)
        x = waveform{i}(:);

        coeffs = mfcc(x, fs, ...
            'NumCoeffs', 40, ...
            'LogEnergy','Replace', ...
            'DeltaWindowLength', 3);  % must be odd

        feats{i} = coeffs'; % size: [40 × T]
    end

    embCell = cell(size(feats));

    % === Forward through network up to 'fc_embed' ===
    for i = 1:numel(feats)
        feat = feats{i};   % [40 × T]
        inputSize = net.Layers(1).InputSize; % [40, maxLen, 1]

        numFeat = inputSize(1);   % 40
        maxLen  = inputSize(2);   % time frames

        % Pad/truncate
        T = size(feat,2);
        featPadded = zeros(numFeat, maxLen, 'single');
        if T < maxLen
            featPadded(:,1:T) = single(feat(:,1:T));
        else
            featPadded = single(feat(:,1:maxLen));
        end

        % 4D input [features × time × 1 × batch]
        X = reshape(featPadded, [numFeat, maxLen, 1, 1]);

        % Extract embeddings
        embCell{i} = activations(net, X, 'fc_embed', 'OutputAs','rows');
    end

    % === Combine and normalize ===
    embRaw = cell2mat(embCell);
    embRaw = double(embRaw);
    emb = embRaw ./ (vecnorm(embRaw,2,2) + eps);

    % === Debug ===
    if all(emb(:) == 0)
        warning('getEmbedding:zeroOutput', ...
            'Embeddings are all zeros. Check fc_embed layer and MFCC params.');
    elseif any(isnan(emb(:)))
        warning('getEmbedding:nanOutput', ...
            'Embeddings contain NaNs. Check input and network stability.');
    end
end
