require 'paths'
require 'rnn'
require 'optim'
local dl = require 'dataload'

--[[
TODO
test rnn:sharedClone
test cuda LSRC
SeqGen (backward, test train/eval)
Bigrams
GARNNReward
feed condition into D()
--]]

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train generative adversarial RNNLM to generate sequences')
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.05, 'learning rate at t=0')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', -1, 'momentum learning factor')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- rnn
cmd:option('--xplogpath', '', 'path to the pretrained RNNLM that is used to initialize the GAN')
cmd:option('--nsample', 100, 'how may words w[t+1] to sample from the bigram distribution given w[t]')
cmd:option('--k', 1, 'number of discriminator updates per generator update')
cmd:option('--ngen', 50, 'number of words generated per update')
cmd:option('--dhiddensize', '{}', 'table of discriminator hidden sizes')
-- data
cmd:option('--batchsize', 32, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('--validsize', -1, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'garnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})
assert(opt.xplogpath ~= '' and paths.filep(opt.xplogpath), "Expecting pre-trained language model at --xplogpath")
opt.dhiddensize = loadstring(" return "..opt.dhiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
opt.inputsize = opt.inputsize == -1 and opt.hiddensize[1] or opt.inputsize
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id
opt.version = 1

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

--[[ data set ]]--

local trainset, validset, testset = dl.loadPTB({opt.batchsize,1,1})
if not opt.silent then 
   print("Vocabulary size : "..#trainset.ivocab) 
   print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())
end

--[[ language model ]]--

local xplog = torch.load(opt.xplogpath)
assert(xplog.dataset == 'PennTreeBank', "GAN-RNNLM currently only supports LMs trained with recurrent-language-model.lua script")
local lm = torch.type(xplog.model) == 'nn.Serial' and lm.modules[1] or lm

print("Loaded language model")
print(lm)

-- clear all sharedClones
lm:clearStepModules()

-- get pre-trained word embedding space
local lookuptable = assert(lm:findModules('nn.Lookuptable')[1], 'Missing LookupTable')

-- get sequencer containing the stepmodule (rnn stack that is applied to every time-step)
local sequencer = assert(lm:findModules('nn.Sequencer')[1], 'Missing Sequencer')
local stepmodule = sequencer:get(1):get(1)
assert(torch.type(stepmodule) == 'nn.Sequential')

-- get output linear layer
local linear = table.remove(stepmodule.modules, #stepmodule.modules-1)
assert(torch.type(linear) == 'nn.Linear')
local softmaxtype = torch.type(table.remove(stepmodule.modules, #stepmodule.modules))
assert(softmaxtype == 'nn.SoftMax' or softmaxtype == 'nn.LogSoftMax')

--[[ generator network : G(z) ]]--

local gsm = stepmodule:sharedClone()
if lm:get(2) == 'nn.Dropout' then
   table.insert(gsm.modules, 1, nn.Dropout())
end
table.insert(gsm.modules, 1, lookup:sharedClone())

-- LSRC requires output of bigrams
local lsrc = nn.LSRC(1,1):fromLinear(linear)

local cachepath = paths.concat(opt.savepath, 'ptb.t7')
local bigram 
if paths.filep(opt.cachepath) then
   bigram = torch.load(bigram)
else
   local bigrams = dl.buildBigrams(trainset)
   bigram = nn.Bigrams(bigrams, opt.nsample)
   torch.save(opt.cachepath, bigram)
end

gsm = nn.Sequential()
   :add(nn.ConcatTable():add(gsm):add(bigram))
   :add(lsrc)

local g_net = nn.SequenceGenerator(gsm, opt.ngen) -- G(z)
print("Generator Network")
print(g_net)

local gupdateGradInput = g_net.updateGradInput
local gaccGradParameters = g_net.accGradParameters
function g_net:doBackward()
   self.updateGradInput = gupdateGradInput
   self.accGradParameters = gaccGradParameters
end


--[[ discriminator network : D(x) ]]--

local dsm = stepmodule:clone() -- the rnns layers of g_net and d_net are not shared
if lm:get(2) == 'nn.Dropout' then
   table.insert(dsm.modules, 1, nn.Dropout())
end
table.insert(dsm.modules, 1, lookup:sharedClone()) -- lookuptables are shared

-- the last hidden state is used to discriminate the entire sequence
local d_net = nn.Sequential() -- D(x)
   :add(nn.Sequencer(dsm))
   :add(nn.Select(1,-1))

local inputsize = lsrc.inputsize
for i, hiddensize in ipairs(opt.dhiddensize)
   d_net:add(nn.Linear(inputsize, hiddensize))
   d_net:add(nn.Tanh())
   inputsize = hiddensize
end

d_net:add(nn.Linear(inputsize, 1))
d_net:add(nn.Sigmoid())

local daccGradParameters = d_net.accGradParameters
function d_net:doBackward()
   self.accGradParameters = daccGradParameters
end
function d_net:dontBackward()
   self.accGradParameters = function() end
end

--[[ discriminator of generative samples : D(G(z)) ]]--

dg_net = nn.Sequential() -- D(G(z))
   :add(g_net)
   :add(d_net:sharedClone())

--[[ loss function ]]--

local g_criterion = nn.GANReward()

-- -log(1-D(G(z)))
local d_criterion = nn.BCECriterion()
local d_target = torch.Tensor()

--[[ CUDA ]]--

if opt.cuda then
   g_net:cuda()
   d_net:cuda()
   criterion:cuda()
   d_criterion:cuda()
   d_target = d_target:cuda()
end

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = 'PennTreeBank'
xplog.vocab = trainset.vocab
-- will only serialize params
xplog.model = nn.Serial(lm); xplog.model:mediumSerial()
xplog.g_net = nn.Serial(g_net); xplog.g_net:mediumSerial()
xplog.d_net = nn.Serial(d_net); xplog.d_net:mediumSerial()
xplog.dg_net = nn.Serial(dg_net); xplog.dg_net:mediumSerial()
xplog.d_criterion = d_criterion
-- keep a log of error for each epoch
xplog.dgerr, xplog.derr = {}, {}
xplog.accuracy, xplog.confusion = {}, {}
xplog.epoch = 0
paths.mkdir(opt.savepath)

-- confusion matrix
function optim.ConfusionMatrix:batchAddBCE(predictions, targets)
   self._bcepred = self._bcepred or predictions.new()
   self._bcepred:gt(predictions, 0.5):add(1)
   self._bcetarg = self._bcetarg or targets.new()
   self._bcetarg:add(targets, 1)
   return self:batchAdd(self._bcepred, self._bcetarg)
end

--[[ training loop ]]--

-- Pg(z) is the unigram distribution
trainset.unigrams:div(trainset.unigrams:sum())
local Pgen = torch.AliasMultinomial(trainset.unigrams)
local z = torch.LongTensor(opt.batchsize)

local epoch = 1

opt.lr = opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
   print("")
   print("Epoch #"..epoch.." :")
   
   local a = torch.Timer()
   dg_net:training()
   d_net:training()
   local sum_dg_err, sum_d_err = 0, 0
   local dg_count, d_count = 0, 0
   local cm = optim.ConfusionMatrix{0,1}
   local k = 0
   for i, inputs, targets in trainset:subiter(opt.ngen, opt.trainsize) do -- x ~ Pdata(x)
      -- 1.1 train discriminator D()
      k = k + 1
      
      -- z ~ Pg(z) : sample some words to condition the generator
      z = Pgen:batchdraw(z)
      
      -- D(G(z)) : forward/backward z through disc. generator network
      dg_net:get(1):dontBackward()
      dg_net:get(2):doBackward()
      dg_net:get(2):zeroGradParameters()
      local dg_output = dg_net:forward(z)
      local dg_err = d_criterion:forward(dg_output)
      sum_dg_err = sum_dg_err + dg_err
      dg_count = dg_count + 1
      
      d_target:resize(opt.batchsize):fill(0)
      local d_gradOutput = d_criterion:backward(dg_output, d_target)
      dg_net:backward(z, dg_gradOutput)
      
      cm:batchAddBCE(dg_output, d_target)
      
      -- D(x) : forward/backward x through discriminator network
      local d_output = d_net:forward(inputs)
      local d_err = d_criterion:forward(d_output, d_target:fill(1))
      sum_d_err = sum_d_err + d_err
      d_count = d_count + 1
      
      local d_gradOutput = d_criterion:backward(d_output, d_target)
      d_net:zeroGradParameters()
      d_net:backward(inputs, d_gradOutput)
      
      cm:batchAddBCE(d_output, d_target)
      
      -- update D(G(z))
      if opt.cutoff > 0 then
         dg_net:get(2):gradParamClip(opt.cutoff)
      end
      dg_net:get(2):updateGradParameters(opt.momentum)
      dg_net:get(2):updateParameters(opt.lr)
      
      -- update D(x)
      if opt.cutoff > 0 then
         d_net:gradParamClip(opt.cutoff)
      end
      d_net:updateGradParameters(opt.momentum)
      d_net:updateParameters(opt.lr)
      
      if k == opt.k then 
         -- train generator G(z)
         k = 0
         
         -- z ~ Pg(z) : sample some words to condition the generator
         z = Pgen:batchdraw(z)
      
         -- D(G(z)) : forward/backward z through disc. generator network
         dg_net:get(1):doBackward()
         dg_net:get(1):zeroGradParameters()
         dg_net:get(2):dontBackward()
         local dg_output = dg_net:forward(z)
         local dg_err = d_criterion:forward(dg_output) -- TODO : reinforce criterion
         sum_dg_err = sum_dg_err + dg_err
         dg_count = dg_count + 1
         
         d_target:resize(opt.batchsize):fill(0)
         local dg_gradOutput = d_criterion:backward(dg_output, d_target)
         dg_net:backward(z, dg_gradOutput)
         
         cm:batchAddBCE(dg_output, d_target)
         
         -- update D(G(z))
         if opt.cutoff > 0 then
            dg_net:get(1):gradParamClip(opt.cutoff)
         end
         dg_net:get(1):updateGradParameters(opt.momentum)
         dg_net:get(1):updateParameters(opt.lr)
         
      end

      if opt.progress then
         xlua.progress(math.min(i + opt.ngen, opt.trainsize), opt.trainsize)
      end

      if i % 1000 == 0 then
         collectgarbage()
      end

   end
   
   -- learning rate decay
   if opt.schedule then
      opt.lr = opt.schedule[epoch] or opt.lr
   else
      opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
   end
   opt.lr = math.max(opt.minlr, opt.lr)
   
   if not opt.silent then
      print("learning rate", opt.lr)
   end

   if cutorch then cutorch.synchronize() end
   local speed = opt.trainsize*opt.batchsize/a:time().real
   print(string.format("Speed : %f words/second; %f ms/word", speed, 1000/speed))

   xplog.dgerr[epoch] = sum_dg_err/dg_count
   xplog.derr[epoch] = sum_d_err/d_count
   print(string.format("Training Err: D(x)=%f, D(G(z))=%f", xplog.derr[epoch], xplog.dgerr[epoch))
   
   print(cm)
   xplog.accuracy[epoch] = cm.totalValids
   xplog.confusion[epoch] = cm

end

print("Evaluate model using : ")
print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and ' --cuda' or ''))
