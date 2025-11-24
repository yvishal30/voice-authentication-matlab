function build_enrollment_embeddings()
% Build average normalized embeddings per speaker
% Uses trained net + getEmbedding (auto loads everything)

    % === Paths ===
    baseDir   = 'E:\VProject\MATLAB\VoiceAuthProject';
    modelsDir = fullfile(baseDir,'models');
    dataDir   = fullfile(baseDir,'dataset_wav','clips');
    outFile   = fullfile(modelsDir,'enrollment.mat');

    % === Check dataset ===
    if ~exist(dataDir,'dir')
        error('Dataset folder not found: %s', dataDir);
    end

    % === Audio datastore ===
    adsTrain = audioDatastore(dataDir, ...
        'IncludeSubfolders', true, ...
        'FileExtensions', '.wav', ...
        'LabelSource', 'foldernames');

    if isempty(adsTrain.Files)
        error('No WAV files found in %s', dataDir);
    end

    proto = struct();

    % === Setup progress bar ===
    nFiles = numel(adsTrain.Files);
    h = waitbar(0, 'Building enrollment embeddings...');

    % === Loop over all files ===
    for i = 1:nFiles
        [y, fs] = audioread(adsTrain.Files{i});
        if size(y,2) > 1, y = mean(y,2); end  % mono
        if fs ~= 16000
            y = resample(y,16000,fs);
            fs = 16000;
        end

        spk = char(adsTrain.Labels(i));

        % Call new getEmbedding (auto-loads net)
        emb = getEmbedding(y, fs);

        if isfield(proto, spk)
            proto.(spk) = [proto.(spk); emb];
        else
            proto.(spk) = emb;
        end

        % Update progress bar
        waitbar(i/nFiles, h, sprintf('Processing file %d of %d...', i, nFiles));
    end

    % Close progress bar
    close(h);

    % === Average embedding per speaker ===
    fn = fieldnames(proto);
    for i = 1:numel(fn)
        M = proto.(fn{i});
        avg = mean(M,1);
        proto.(fn{i}) = avg ./ (norm(avg,2) + eps);
    end

    % === Save enrollment embeddings ===
    save(outFile,'proto');
    fprintf('✅ Enrollment embeddings saved to %s\n', outFile);
end
