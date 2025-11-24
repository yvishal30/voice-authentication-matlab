function convert_all_to_wav()
% Convert ALL mp3 in dataset/en/clips to 16kHz mono wav
% Stores output in dataset_wav/clips

rootFolder = 'C:\Users\Vishal\OneDrive\Documents\MATLAB\VoiceAuthProject';
mp3Dir     = fullfile(rootFolder, 'dataset', 'en', 'clips');
wavDir     = fullfile(rootFolder, 'dataset_wav', 'clips');
targetFs   = 16000;

if ~exist(wavDir,'dir')
    mkdir(wavDir);
end

files = dir(fullfile(mp3Dir, '*.mp3'));
fprintf("🎵 Found %d mp3 files. Converting...\n", numel(files));

for k = 1:numel(files)
    src = fullfile(files(k).folder, files(k).name);
    [~, name, ~] = fileparts(files(k).name);
    outFile = fullfile(wavDir, [name '.wav']);

    % Skip if already converted
    if exist(outFile,'file')
        continue;
    end

    try
        [y, fs] = audioread(src);
        if size(y,2) > 1
            y = mean(y,2); % convert stereo → mono
        end
        if fs ~= targetFs
            y = resample(y, targetFs, fs); % resample to 16k
        end
        audiowrite(outFile, y, targetFs);
    catch ME
        fprintf("⚠️ Skipping %s (error: %s)\n", src, ME.message);
    end

    if mod(k,1000)==0
        fprintf("✅ Converted %d/%d files\n", k, numel(files));
    end
end

disp('🎯 Conversion of full dataset complete!');
end
