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
std reward
D() reward at each time-step

seqgen shouldn't receive gradients for LSRC on conditions

issue:
generator gets stuck in local minima by REINFORCE
solutions:
e-greedy, temperature, train LM
condition on trainset samples
decrease size of training set samples when generator gets better.

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
cmd:option('--fixgen', false, 'fix the generator (dont let it learn)')
cmd:option('--epsilon', 0, 'epsilon greedy value defaults to 0.1/nsample')
cmd:option('--savefreq', 3, 'save model every savefreq epochs')
cmd:option('--traincond', false, 'use the training set to condition the G(z), i.e. z~P_data(x)')
cmd:option('--smoothtarget', false, 'discriminator targets are 0.1 and 0.9 instead of 0 and 1')
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
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id
opt.version = 4 -- trainset is used as z; disc. sees cond and gen: D(z..G(z))
opt.epsilon = opt.epsilon == -1 and 0.1/opt.nsample or opt.epsilon
if not opt.silent then
   table.print(opt)
end

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
   bigram = nn.Bigrams(bigrams, opt.nsample, opt.epsilon)
   torch.save(opt.cachepath, bigram)
end
print("Mean bigram size: "..bigram:statistics())

if opt.cuda then
   bigram = nn.DontCast(bigram, true, true, 'torch.LongTensor')
end

gsm = nn.Sequential()
   :add(nn.ConcatTable():add(gsm):add(bigram))
   :add(lsrc)

local seqgen = nn.SequenceGenerator(gsm, opt.ngen)
local g_net = nn.Sequential() -- G(z)
   :add(nn.Convert())
   :add(seqgen)
   
print("Generator Network: G(z)")
print(g_net)
print""

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

print("Discriminator Network: D(x)")
print(d_net)
print""

if opt.cuda then
   d_net:cuda()
   g_net:cuda()
end

--[[ discriminator of generative samples : D(G(z)) ]]--

local dg_net = d_net:sharedClone()

local daccGradParameters = dg_net.accGradParameters
function dg_net:doBackward()
   self.accGradParameters = daccGradParameters
end
function dg_net:dontBackward()
   self.accGradParameters = function() end
end
   
--[[ loss function ]]--

local g_criterion = nn.BinaryClassReward(g_net, opt.rewardscale)
-- add the baseline reward predictor
local basereward = nn.Add(1)
local b_zero = torch.zeros(opt.batchsize, 1)

-- -log(1-D(G(z)))
local d_criterion = nn.BCECriterion()
local d_target = torch.Tensor()

--[[ CUDA ]]--

if opt.cuda then
   d_criterion:cuda()
   g_criterion:cuda()
   d_target = d_target:cuda()
   basereward:cuda()
   b_zero = b_zero:cuda()
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

-- z ~ Pg(z) : sample some words to condition the generator
local function drawPgen(z)
   if opt.traincond then
      opt.ncond = opt.ncond or opt.ngen
      local cond = trainset.data:narrow(1, math.random(1, trainset:size(1)-opt.ncond), opt.ncond)
      z:resize(opt.ncond, opt.batchsize):copy(cond)
      seqgen.ngen = opt.ngen-opt.ncond+1
   else
      z = Pgen:batchdraw(z)
   end
   return z
end

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
   g_net:training()
   local sum_dg_err, sum_d_err, sum_g_err = 0.0000001, 0.0000001, 0.0000001
   local dg_count, d_count, g_count = 0, 0, 0
   local cm = optim.ConfusionMatrix{0,1}
   
   for i, inputs, targets in trainset:subiter(opt.ngen+1, opt.trainsize) do -- x ~ Pdata(x)
      
      if opt.fixgen or evalcount <= 0 or math.random() > p_train_gen then
         -- train or evaluate discriminator D()
         
         z = drawPgen(z)
         
         -- D(G(z)) : forward/backward z through disc. generator network
         local g_output = g_net:forward(z)
         
         -- input to discriminator is concatenation of z and g_output
         d_input = d_input or g_output.new()
         d_input:resize(g_output:size(1)+z:size(1), opt.batchsize)
         d_input:narrow(1,1,z:size(1)):copy(z)
         d_input:narrow(1,z:size(1)+1,g_output:size(1)):copy(g_output)
         
         dg_net:training()
         dg_net:forget()
         local dg_output = dg_net:forward(d_input)
         
         d_target:resize(opt.batchsize):fill(opt.smoothtarget and 0.1 or 0)
         local dg_err = d_criterion:forward(dg_output, d_target)
         
         if evalcount <= 0 then
            d_target:fill(0)
            cm:batchAddBCE(dg_output, d_target)
         else
            sum_dg_err = sum_dg_err + dg_err
            dg_count = dg_count + 1
            
            local gradOutput = d_criterion:backward(dg_output, d_target)
            dg_net:zeroGradParameters()
            dg_net:doBackward()
            dg_net:backward(d_input, gradOutput)
         end
         
         -- D(x) : forward/backward x through discriminator network
         local d_output = d_net:forward(inputs)
         
         d_target:fill(opt.smoothtarget and 0.9 or 1)
         local d_err = d_criterion:forward(d_output, d_target)
         
         
         if evalcount <= 0 then
            d_target:fill(1)
            cm:batchAddBCE(d_output, d_target)
            evalcount = opt.evalfreq + 1
            cm:updateValids()
            -- prob of training G() instead of D() is proportional to train accuracy
            p_train_gen = 2*math.max(0, cm.totalValid - 0.5)
         else
            sum_d_err = sum_d_err + d_err
            d_count = d_count + 1
            
            -- backward D(x)
            local gradOutput = d_criterion:backward(d_output, d_target)
            d_net:backward(inputs, gradOutput)
            
            -- update D(x) and D(G(z))
            if opt.cutoff > 0 then
               d_net:gradParamClip(opt.cutoff)
            end
            d_net:updateGradParameters(opt.momentum)
            d_net:updateParameters(opt.lr)
            
            if not tested_grad_d then
               local p, gp = d_net:parameters()
               local p2, gp2 = dg_net:parameters()
               for i=1,#p do
                  assert(math.abs(p[i]:sum() - p2[i]:sum()) < 0.000001)
                  assert(math.abs(gp[i]:sum() - gp2[i]:sum()) < 0.000001)
               end
               print"TESTED"
               tested_grad_d = true
            end
         end
         
      else
         -- train generator G(z)
         
         -- z ~ Pg(z) : sample some words to condition the generator
         z = drawPgen(z)
      
         -- D(G(z)) : forward/backward z through disc. generator network
         local g_output = g_net:forward(z)
          
         -- input to discriminator is concatenation of z and g_output
         d_input = d_input or g_output.new()
         d_input:resize(g_output:size(1)+z:size(1), opt.batchsize)
         d_input:narrow(1,1,z:size(1)):copy(z)
         d_input:narrow(1,z:size(1)+1,g_output:size(1)):copy(g_output)
         
         local dg_output = dg_net:forward(d_input)[{{},1}]
         
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
            g_net:reinforce(xplog.mse.vrReward)  
            
            -- zero gradInput (this criterion has no gradInput for class pred)
            xplog.mse._gradZero = xplog.mse._gradZero or dg_output.new()
            xplog.mse._gradZero:resizeAs(dg_output):zero()
            gradOutput = {}
            gradOutput[1] = xplog.mse._gradZero
            
            -- learn the baseline reward
            xplog.mse:forward(br, xplog.mse.reward)
            gradOutput[2] = xplog.mse:backward(br, xplog.mse.reward)
         else
            -- reward=1 for classifying gen. sample. as training data
            d_target:fill(1) 
            g_err = g_criterion:forward({dg_output, br}, d_target) 
            gradOutput = g_criterion:backward({dg_output, br}, d_target)
         end
         
         sum_g_err = sum_g_err + g_err
         g_count = g_count + 1
   
         g_net:zeroGradParameters()
         g_net:backward(z, g_output) -- g_output is ignored
         basereward:zeroGradParameters()
         basereward:backward(b_zero, gradOutput[2])
         
         
         -- update D(G(z))
         if opt.cutoff > 0 then
            g_net:gradParamClip(opt.cutoff)
         end
         g_net:updateGradParameters(opt.momentum)
         g_net:updateParameters(opt.lr)
         
         -- update baseline reward
         basereward:updateGradParameters(opt.momentum)
         basereward:updateParameters(opt.lr)
         
      end
      
      evalcount = evalcount - 1

      if opt.progress then
         xlua.progress(math.min(i + opt.ngen + 1, opt.trainsize), opt.trainsize)
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
   
   if opt.ncond and cm.totalValid < 0.6 then
      opt.ncond = math.max(1, opt.ncond - 1)
      print("Decreasing size of condition z to "..opt.ncond)
   end
   -- draw a sample
   g_net:evaluate()

   local given = 'never'
   local sampletext = {given}
   seqgen.ngen = opt.ngen
   local input = opt.cuda and torch.CudaTensor(1,1) or torch.LongTensor(1,1) -- seqlen x batchsize
   input[{1,1}] = trainset.vocab[given]
   local output = g_net:forward(input)

   for i=1,output:size(1) do
      table.insert(sampletext, trainset.ivocab[output[i][1]])
   end
   print"generated sample:"
   print(table.concat(sampletext, ' '))

   if epoch % opt.savefreq == 0 then
      print("saving model at "..paths.concat(opt.savepath, opt.id..'.t7'))
      torch.save(paths.concat(opt.savepath, opt.id..'.t7'), xplog)
   end
   epoch = epoch + 1
end

print("Evaluate model using : ")
print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and ' --cuda' or ''))
