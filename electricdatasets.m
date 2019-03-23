delete(gcp("nocreate"))
pool=parpool;
[basename, filepath] = uigetfile('Electricdata.csv', 'Select a dataset');
filename = fullfile(filepath, basename);
data = csvread(filename, 1);
if isnumeric(filename)
   uiwait(msgbox('User cancelled'));
   return
end

inputs= data(:,1:12);
outputs= data(:,13);


inputs=inputs';
outputs=outputs';
x = inputs;
t = outputs;
trainFcn = 'trainscg';  
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);


net.input.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand'; 
net.divideMode = 'sample';  
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


net.performFcn = 'crossentropy';  


net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotconfusion', 'plotroc'};
[net,tr] = train(net,x,t);


y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);


trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);
view(net)
% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

if (false)

  genFunction(net,'myNeuralNetworkFunction');
  y = myNeuralNetworkFunction(x);
end
if (false)
  genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
  y = myNeuralNetworkFunction(x);
end
if (false)
  gensim(net);
end
