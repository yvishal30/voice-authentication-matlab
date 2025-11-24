function select_subset()
% Extract 500MB subset from dataset_raw into dataset_subset

DATA_ROOT = 'C:\Users\Vishal\OneDrive\Documents\MATLAB\VoiceAuthProject\dataset\en';
OUT_ROOT  = 'C:\Users\Vishal\OneDrive\Documents\MATLAB\VoiceAuthProject\dataset_subset';
metaTSV   = fullfile(DATA_ROOT, 'validated.tsv');
TARGET_MB = 500;

if ~exist(OUT_ROOT,'dir'), mkdir(OUT_ROOT); end
clipsDir = fullfile(DATA_ROOT,'clips');
targetBytes = TARGET_MB * 1024^2;
audioFiles = dir(fullfile(clipsDir,'*.mp3'));

meta = readtable(metaTSV,'FileType','text','Delimiter','\t','ReadVariableNames',true);

rng(0);
shuffled = audioFiles(randperm(numel(audioFiles)));
cumBytes = 0; copied = 0;

for k=1:numel(shuffled)
    src = fullfile(shuffled(k).folder, shuffled(k).name);
    destDir = fullfile(OUT_ROOT,'clips');
    if ~exist(destDir,'dir'), mkdir(destDir); end
    dest = fullfile(destDir, shuffled(k).name);
    copyfile(src,dest);
    fbytes = shuffled(k).bytes;
    cumBytes = cumBytes + fbytes;
    copied = copied + 1;
    if cumBytes >= targetBytes
        fprintf('Subset ready: %d files, %.2f MB\n',copied,cumBytes/1024^2);
        break;
    end
end
end
