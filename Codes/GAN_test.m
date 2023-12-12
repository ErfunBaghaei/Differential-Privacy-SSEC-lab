clc; close all; clear all;
%untar('flower_photos.tgz','dataset')
%unzip('archive.zip','cats_dataset')
datasetFolder = fullfile('one-dataset');%imageFolder);
datasetFolder1 = fullfile('zero-dataset');


%trainImagesFile = fullfile('train-images-idx3-ubyte');
%trainImagesFile = 'train-images-idx3-ubyte.gz';
%XTrain = processImagesMNIST(trainImagesFile);
% data = load('data');
%  for i=501:1000
%     im = reshape(data.X(i,:),[20,20]);
%     baseFileName = sprintf('Image #%d.png', i-500);
%     fullFileName = fullfile('one-dataset', baseFileName);
%     imwrite(im, fullFileName);
% end                                                                   
                                  
                                 
                                                                           
imds = imageDatastore(datasetFolder, ...                                     
    'IncludeSubfolders',true);
augmenter = imageDataAugmenter('RandXReflection',true);
augimds = augmentedImageDatastore([20 20],imds,'DataAugmentation',augmenter); 


imds1 = imageDatastore(datasetFolder1, ...
    'IncludeSubfolders',true);
augmenter1 = imageDataAugmenter('RandXReflection',true);
augimds1 = augmentedImageDatastore([20 20],imds1,'DataAugmentation',augmenter1);



filterSize = 5;
numFilters = 64;
numLatentInputs = 100;

projectionSize = [6 6 512];

layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    transposedConv2dLayer(filterSize,4*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bnorm1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',1,'Cropping','same','Name','tconv2')
    batchNormalizationLayer('Name','bnorm2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,numFilters,'Stride',1,'Cropping','same','Name','tconv3')
    batchNormalizationLayer('Name','bnorm3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping','same','Name','tconv4')
    tanhLayer('Name','tanh')];
lgraphGenerator = layerGraph(layersGenerator);


dlnetGenerator = dlnetwork(lgraphGenerator);

% cats generator
layersGenerator1 = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    transposedConv2dLayer(filterSize,4*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bnorm1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',1,'Cropping','same','Name','tconv2')
    batchNormalizationLayer('Name','bnorm2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,numFilters,'Stride',1,'Cropping','same','Name','tconv3')
    batchNormalizationLayer('Name','bnorm3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping','same','Name','tconv4')
    tanhLayer('Name','tanh')];

lgraphGenerator1 = layerGraph(layersGenerator1);


dlnetGenerator1 = dlnetwork(lgraphGenerator1);

%--------------



dropoutProb = 0.5;
numFilters = 64;
scale = 0.2;

inputSize = [20 20 1];
filterSize = 5;

layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    dropoutLayer(0.5,'Name','dropout')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding','same','Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(2,1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

%cats discriminator
layersDiscriminator1 = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    dropoutLayer(0.5,'Name','dropout')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding','same','Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(2,1,'Name','conv5')];

lgraphDiscriminator1 = layerGraph(layersDiscriminator1);
dlnetDiscriminator1 = dlnetwork(lgraphDiscriminator1);


%------------1



numEpochs = 500;
miniBatchSize = 100;


learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;



flipFactor = 0.3;
validationFrequency = 10;



augimds.MiniBatchSize = miniBatchSize;
augimds1.MiniBatchSize = miniBatchSize;

executionEnvironment = "auto";

mbq = minibatchqueue(augimds,...
    'MiniBatchSize',miniBatchSize,...
    'PartialMiniBatch','discard',...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat','SSCB',...
    'OutputEnvironment',executionEnvironment);

mbq1 = minibatchqueue(augimds1,...
    'MiniBatchSize',miniBatchSize,...
    'PartialMiniBatch','discard',...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat','SSCB',...
    'OutputEnvironment',executionEnvironment);


trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];



trailingAvgGenerator1 = [];
trailingAvgSqGenerator1 = [];
trailingAvgDiscriminator1 = [];
trailingAvgSqDiscriminator1 = [];



numValidationImages = 100;
ZValidation = randn(1,1,numLatentInputs,numValidationImages,'single');

ZValidation1 = randn(1,1,numLatentInputs,numValidationImages,'single');


dlZValidation = dlarray(ZValidation,'SSCB');
dlZValidation1 = dlarray(ZValidation1,'SSCB');



if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation1 = gpuArray(dlZValidation1);
end

f = figure;
f.Position(3) = 2*f.Position(3);


imageAxes = subplot(2,2,1);
scoreAxes = subplot(2,2,2);


imageAxes1 = subplot(2,2,3);
scoreAxes1 = subplot(2,2,4);



lineScoreGenerator = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes, 'Color', [0.85 0.325 0.098]);
legend('Generator','Discriminator');
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on

lineScoreGenerator1 = animatedline(scoreAxes1,'Color',[0 0.447 0.741]);
lineScoreDiscriminator1 = animatedline(scoreAxes1, 'Color', [0.85 0.325 0.098]);

legend('Generator','Discriminator');
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on







iteration = 0;
endflag = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    

    
    % Reset and shuffle datastore.
    shuffle(mbq);
    shuffle(mbq1);

    % Loop over mini-batches.
    while ((hasdata(mbq))&&(hasdata(mbq1)))
        iteration = iteration + 1;
        
     
             
    %transfering dataset
    %display(endflag);
      %  Read mini-batch of data.
       if (mod(fix(epoch/10),2) == 0 || epoch<100)
        dlX = next(mbq);
        dlX1 = next(mbq1);
        dlnetGenerator_test = dlnetGenerator;
        dlnetGenerator1_test = dlnetGenerator1;
       else 
           endflag = endflag + 1;
           dlX =  dlXGeneratedValidation1_test;
            dlX1 = dlXGeneratedValidation_test;

       end
       
       
       if (endflag == 5 )
           endflag = 0;
           break;
       end
    
    %-----------------------------------
        
        
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'SSCB' (spatial,
        % spatial, channel, batch). If training on a GPU, then convert
        % latent inputs to gpuArray.
        Z = randn(1,1,numLatentInputs,size(dlX,4),'single');
        dlZ = dlarray(Z,'SSCB');  
        
        Z1 = randn(1,1,numLatentInputs,size(dlX1,4),'single');
        dlZ1 = dlarray(Z1,'SSCB');  
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
        end
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ1 = gpuArray(dlZ1);
        end
        
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
        dlnetGenerator.State = stateGenerator;
        
        
        [gradientsGenerator1, gradientsDiscriminator1, stateGenerator1, scoreGenerator1, scoreDiscriminator1] = ...
            dlfeval(@modelGradients, dlnetGenerator1, dlnetDiscriminator1, dlX1, dlZ1, flipFactor);
        dlnetGenerator1.State = stateGenerator1;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        
        [dlnetDiscriminator1,trailingAvgDiscriminator1,trailingAvgSqDiscriminator1] = ...
            adamupdate(dlnetDiscriminator1, gradientsDiscriminator1, ...
            trailingAvgDiscriminator1, trailingAvgSqDiscriminator1, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        
        [dlnetGenerator1,trailingAvgGenerator1,trailingAvgSqGenerator1] = ...
            adamupdate(dlnetGenerator1, gradientsGenerator1, ...
            trailingAvgGenerator1, trailingAvgSqGenerator1, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input
         % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            dlXGeneratedValidation_test = predict(dlnetGenerator_test,dlZValidation);
            
            
            dlXGeneratedValidation1 = predict(dlnetGenerator1,dlZValidation1);
            
            dlXGeneratedValidation1_test = predict(dlnetGenerator1_test,dlZValidation1);
            
        if mod(iteration,validationFrequency) == 0 || iteration == 1
           
            
            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            
            
            I1 = imtile(extractdata(dlXGeneratedValidation1));
            %I1 = extractdata(dlXGeneratedValidation1);
            I1 = rescale(I1);
            
            % Display the images.
            subplot(2,2,1);
            imshow(I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
            
            subplot(2,2,3);
            imshow(I1)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
        end
        
        % Update the scores plot
        subplot(2,2,2)
        addpoints(lineScoreGenerator,iteration,...
            double(gather(extractdata(scoreGenerator))));
        
        addpoints(lineScoreDiscriminator,iteration,...
            double(gather(extractdata(scoreDiscriminator))));
        
        
        
        subplot(2,2,4)
        addpoints(lineScoreGenerator1,iteration,...
            double(gather(extractdata(scoreGenerator1))));
        
        addpoints(lineScoreDiscriminator1,iteration,...
            double(gather(extractdata(scoreDiscriminator1))));
        
        
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
    end
end





ZNew = randn(1,1,numLatentInputs,25,'single');
dlZNew = dlarray(ZNew,'SSCB');


ZNew1 = randn(1,1,numLatentInputs,25,'single');
dlZNew1 = dlarray(ZNew1,'SSCB');


if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
end



if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew1 = gpuArray(dlZNew1);
end



dlXGeneratedNew = predict(dlnetGenerator,dlZNew);


dlXGeneratedNew1 = predict(dlnetGenerator1,dlZNew1);





I = imtile(extractdata(dlXGeneratedNew));
I = rescale(I);

I1 = imtile(extractdata(dlXGeneratedNew1));
I1 = rescale(I1);


f1 = figure;
imshow(I)
axis off
title("Generated Images")
print(f1,'C:\Users\khalaj\Desktop/f1.pdf','-dpdf')


f2 = figure;
imshow(I1)
axis off
title("Generated Images")
print(f2,'C:\Users\khalaj\Desktop/f2.pdf','-dpdf')



function [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor)

% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);
% Convert the discriminator outputs to probabilities.
%dlYPredGenerated
probGenerated = sigmoid(dlYPredGenerated);
probReal = sigmoid(dlYPred);

% Calculate the score of the discriminator.
scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

% Calculate the score of the generator.
scoreGenerator = mean(probGenerated);

% Randomly flip a fraction of the labels of the real images.
numObservations = size(probReal,2);
idx = randperm(numObservations,floor(flipFactor * numObservations));

% Flip the labels
probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

% Calculate the GAN loss.
[lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated);


% For each network, calculate the gradients with respect to the loss.
gradientsGenerator =  dlgradient(dlarray(lossGenerator),dlnetGenerator.Learnables);%,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);



end


function [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated)

% Calculate the loss for the discriminator network.
lossDiscriminator = -mean(log(probReal)) -mean(log(1-probGenerated));

% Calculate the loss for the generator network.
lossGenerator = -mean(log(probGenerated));

end



function X = preprocessMiniBatch(data)
    % Concatenate mini-batch
    X = cat(4,data{:});
    
    % Rescale the images in the range [-1 1].
    X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end










