require 'paths'
require 'rnn'
require 'optim'
require 'nngraph'
local dl = require 'dataload'

-- the new nn implements backward for Sequential, lets use the old one 
-- which just calls updateGradInput/accGradParameters
nn.Sequential.backward = nil

-- backward comp Lua 5.2 optim.ConfusionMatrix
math.log10 = function(x) return math.log(x, 10) end

--[[
TODO
feed condition into D()
std reward
D() reward at each time-step
savefreq 
print samples
--]]

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train generative adversarial RNNLM to generate sequences')
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.1, 'learning rate at t=0')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', -1, 'momentum learning factor')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- rnn
cmd:option('--xplogpath', '', 'path to the pretrained RNNLM that is used to initialize the GAN')
cmd:option('--nsample', 50, 'how may words w[t+1] to sample from the bigram distribution given w[t]')
cmd:option('--ngen', 50, 'number of words generated per update')
cmd:option('--dhiddensize', '{}', 'table of discriminator hidden sizes')
cmd:option('--evalfreq', 10, 'how many updates between evaluations')
cmd:option('--updatelookup', false, 'set to true to enable lookup table updates')
cmd:option('--dreward', false, 'reward = D(G(z))')
cmd:option('--rewardscale', 1, 'the scale of the reward.')
-- data
cmd:option('--batchsize', 32, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
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
opt.version = 2 -- prob of training gen is proportional to disc accuracy

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
local lm = torch.type(xplog.model) == 'nn.Serial' and xplog.model.modules[1] or xplog.model

print("Loaded language model")
print(lm)
print""

-- clear all sharedClones
lm:clearStepModules()

-- get pre-trained word embedding space
local lookuptable = assert(lm:findModules('nn.LookupTable')[1], 'Missing LookupTable')
if not opt.updatelookup then
   lookuptable.accGradParameters = function() end
end

-- get sequencer containing the stepmodule (rnn stack that is applied to every time-step)
local sequencer = assert(lm:findModules('nn.Sequencer')[1], 'Missing Sequencer')
local stepmodule = sequencer:get(1):get(1)
assert(torch.type(stepmodule) == 'nn.Sequential')

-- get output linear layer
local linear = table.remove(stepmodule.modules, #stepmodule.modules-1)
assert(torch.type(linear) == 'nn.Linear')
local softmaxtype = torch.type(table.remove(stepmodule.modules, #stepmodule.modules))
assert(softmaxtype == 'nn.SoftMax' or softmaxtype == 'nn.LogSoftMax')

if opt.cuda then
   lookuptable:cuda()
   stepmodule:cuda()
   linear:cuda()
end


--[[ generator network : G(z) ]]--


local gsm = stepmodule:sharedClone()
if lm:get(2) == 'nn.Dropout' then
   table.insert(gsm.modules, 1, nn.Dropout())
end
table.insert(gsm.modules, 1, lookuptable:sharedClone())

-- LSRC requires output of bigrams
local lsrc = nn.LSRC(1,1):fromLinear(linear)

opt.cachepath = paths.concat(opt.savepath, 'ptb.t7')
local bigram 
if paths.filep(opt.cachepath) then
   bigram = torch.load(opt.cachepath)
else
   local bigrams = dl.buildBigrams(trainset)
   bigram = nn.Bigrams(bigrams, opt.nsample)
   torch.save(opt.cachepath, bigram)
end
print("Mean bigram size: "..bigram:statistics())

if opt.cuda then
   bigram = nn.DontCast(bigram, true, true, 'torch.LongTensor')
end

gsm = nn.Sequential()
   :add(nn.ConcatTable():add(gsm):add(bigram))
   :add(lsrc)

local g_net = nn.Sequential() -- G(z)
   :add(nn.Convert())
   :add(nn.SequenceGenerator(gsm, opt.ngen))
   
print("Generator Network: G(z)")
print(g_net)
print""

local gupdateGradInput = g_net.updateGradInput
local gaccGradParameters = g_net.accGradParameters
function g_net:doBackward()
   self.updateGradInput = gupdateGradInput
   self.accGradParameters = gaccGradParameters
end
function g_net:dontBackward()
   self.updateGradInput = function() end
   self.accGradParameters = function() end
   self.accUpdateGradParameters = function() end
   return self
end


--[[ discriminator network : D(x) ]]--


local dsm = stepmodule:clone() -- the rnns layers of g_net and d_net are not shared
if lm:get(2) == 'nn.Dropout' then
   table.insert(dsm.modules, 1, nn.Dropout())
end
table.insert(dsm.modules, 1, lookuptable:sharedClone()) -- lookuptables are shared

-- the last hidden state is used to discriminate the entire sequence
local d_net = nn.Sequential() -- D(x)
   :add(nn.Convert())
   :add(nn.Sequencer(dsm))
   :add(nn.Select(1,-1))

local inputsize = lsrc.inputsize
for i, hiddensize in ipairs(opt.dhiddensize) do
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

print("Discriminator Network: D(x)")
print(d_net)
print""


--[[ discriminator of generative samples : D(G(z)) ]]--

local _d_net = d_net:sharedClone()
local dg_net = nn.Sequential() -- D(G(z))
   :add(g_net)
   :add(_d_net)
   
local _daccGradParameters = _d_net.accGradParameters
function _d_net:doBackward()
   self.accGradParameters = _daccGradParameters
end
function _d_net:dontBackward()
   self.accGradParameters = function() end
end
   
print("Disc. Generator Network: D(G(z))")
print(dg_net)
print""

--[[ loss function ]]--

local g_criterion = nn.BinaryClassReward(dg_net, opt.rewardscale)
-- add the baseline reward predictor
local basereward = nn.Add(1)
local b_zero = torch.zeros(opt.batchsize, 1)

-- -log(1-D(G(z)))
local d_criterion = nn.BCECriterion()
local d_target = torch.Tensor()

--[[ CUDA ]]--

if opt.cuda then
   a = torch.Timer()
   d_net:cuda()
   dg_net:cuda()
   d_criterion:cuda()
   g_criterion:cuda()
   d_target = d_target:cuda()
   basereward:cuda()
   b_zero = b_zero:cuda()
   print("converted to cuda in "..a:time().real.."s")
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
xplog.basereward = basereward
xplog.d_criterion, xplog.g_criterion = d_criterion, g_criterion
-- keep a log of error for each epoch
xplog.dgerr, xplog.derr, xplog.gerr = {}, {}, {}
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
trainset.unigram = torch.Tensor(#trainset.ivocab)
for i,word in ipairs(trainset.ivocab) do
   trainset.unigram[i] = trainset.wordfreq[word]
end
trainset.unigram:div(trainset.unigram:sum())
local Pgen = torch.AliasMultinomial(trainset.unigram)
local z = torch.LongTensor(1, opt.batchsize)

local epoch = 1
local evalcount = 0 
local p_train_gen = 0.5 -- prob of training generator instead of discriminator

opt.lr = opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
   print("")
   print("Epoch #"..epoch.." :")
   
   local a = torch.Timer()
   dg_net:training()
   d_net:training()
   local sum_dg_err, sum_d_err, sum_g_err = 0.0000001, 0.0000001, 0.0000001
   local dg_count, d_count, g_count = 0, 0, 0
   local cm = optim.ConfusionMatrix{0,1}
   

   for i, inputs, targets in trainset:subiter(opt.ngen, opt.trainsize) do -- x ~ Pdata(x)
      
      if evalcount <= 0 or math.random() > p_train_gen then
         -- train or evaluate discriminator D()
         
         -- z ~ Pg(z) : sample some words to condition the generator
         z = Pgen:batchdraw(z)
         
         -- D(G(z)) : forward/backward z through disc. generator network
         dg_net:get(1):dontBackward()
         dg_net:get(2):doBackward()
         dg_net:get(2):zeroGradParameters()
         local dg_output = dg_net:forward(z)
         
         d_target:resize(opt.batchsize):fill(0)
         local dg_err = d_criterion:forward(dg_output, d_target)
         
         if evalcount <= 0 then
            cm:batchAddBCE(dg_output, d_target)
         else
            sum_dg_err = sum_dg_err + dg_err
            dg_count = dg_count + 1
            
            local gradOutput = d_criterion:backward(dg_output, d_target)
            dg_net:backward(z, gradOutput)
         end
         
         -- D(x) : forward/backward x through discriminator network
         local d_output = d_net:forward(inputs)
         
         d_target:fill(1)
         local d_err = d_criterion:forward(d_output, d_target)
         
         
         if evalcount <= 0 then
            cm:batchAddBCE(d_output, d_target)
            evalcount = opt.evalfreq + 1
            cm:updateValids()
            -- prob of training G() instead of D() is proportional to train accuracy
            p_train_gen = 2*math.max(0, cm.totalValid - 0.5)
         else
            sum_d_err = sum_d_err + d_err
            d_count = d_count + 1
            
            local gradOutput = d_criterion:backward(d_output, d_target)
            d_net:zeroGradParameters()
            d_net:backward(inputs, gradOutput)
            
            
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
         end
         
      else
         -- train generator G(z)
         
         -- z ~ Pg(z) : sample some words to condition the generator
         z = Pgen:batchdraw(z)
      
         -- D(G(z)) : forward/backward z through disc. generator network
         dg_net:get(1):doBackward()
         dg_net:get(1):zeroGradParameters()
         dg_net:get(2):dontBackward()
         local dg_output = dg_net:forward(z)[{{},1}]
         
         -- get baseline reward for REINFORCE criterion
         local br = basereward:forward(b_zero)
         -- reward G(z) when D(G(z)) gets fooled into thinking it sees D(x)
         
         local gradOutput, g_err
         if opt.dreward then
            -- forward
            xplog.mse = xplog.mse or g_criterion.criterion
            xplog.mse.reward = xplog.mse.reward or dg_output.new()
            -- reward = D(G(z)) (the more G(z) can fool D into thinking it is x, the better the reward)
            xplog.mse.reward:resize(opt.batchsize):copy(dg_output)
            xplog.mse.reward:mul(opt.rewardscale)
            
            -- loss = -sum(reward)
            g_err = -xplog.mse.reward:sum()/opt.batchsize
            
             -- reduce variance of reward using baseline
            xplog.mse.vrReward = xplog.mse.vrReward or xplog.mse.reward.new()
            xplog.mse.vrReward:resizeAs(xplog.mse.reward):copy(xplog.mse.reward)
            xplog.mse.vrReward:add(-1, br)
            xplog.mse.vrReward:div(opt.batchsize)
            
            -- broadcast reward to modules
            dg_net:reinforce(xplog.mse.vrReward)  
            
            -- zero gradInput (this criterion has no gradInput for class pred)
            xplog.mse._gradZero = xplog.mse._gradZero or dg_output.new()
            xplog.mse._gradZero:resizeAs(dg_output):zero()
            gradOutput = {}
            gradOutput[1] = xplog.mse._gradZero
            
            -- learn the baseline reward
            xplog.mse:forward(br, xplog.mse.reward)
            gradOutput[2] = xplog.mse:backward(br, xplog.mse.reward)
         else
            -- reward=1 for classifying gen. sample. as training data, i.e.
            d_target:fill(1) 
            g_err = g_criterion:forward({dg_output, br}, d_target) 
            gradOutput = g_criterion:backward({dg_output, br}, d_target)
         end
         
         sum_g_err = sum_g_err + g_err
         g_count = g_count + 1
   
         dg_net:zeroGradParameters()
         dg_net:backward(z, gradOutput[1])
         basereward:zeroGradParameters()
         basereward:backward(b_zero, gradOutput[2])
         
         d_target:resize(opt.batchsize):fill(0)
         
         -- update D(G(z))
         if opt.cutoff > 0 then
            dg_net:get(1):gradParamClip(opt.cutoff)
         end
         dg_net:get(1):updateGradParameters(opt.momentum)
         dg_net:get(1):updateParameters(opt.lr)
         
         -- update baseline reward
         basereward:updateGradParameters(opt.momentum)
         basereward:updateParameters(opt.lr)
         
      end
      
      evalcount = evalcount - 1

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
   print(string.format("Speed : %f train-words/second; %f ms/word", speed, 1000/speed))

   xplog.epoch = epoch
   xplog.dgerr[epoch] = sum_dg_err/dg_count
   xplog.derr[epoch] = sum_d_err/d_count
   xplog.gerr[epoch] = sum_g_err/g_count
   print(string.format("Loss: D(x)=%f, D(G(z))=%f; nupdate=%d", xplog.derr[epoch], xplog.dgerr[epoch], dg_count))
   print(string.format("Reward: G(z)=%f; nupdate=%d", -xplog.gerr[epoch], g_count))
   
   print(cm)
   xplog.accuracy[epoch] = cm.totalValid
   xplog.confusion[epoch] = cm

   print("saving model at "..paths.concat(opt.savepath, opt.id..'.t7'))
   torch.save(paths.concat(opt.savepath, opt.id..'.t7'), xplog)
   epoch = epoch + 1
end

print("Evaluate model using : ")
print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and ' --cuda' or ''))
