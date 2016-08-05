require 'nngraph'
require 'rnn'
require 'optim'
local dl = require 'dataload'


--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a RNNLM')
cmd:text('Options:')
cmd:option('--xplogpath', '', 'path to a previously saved xplog containing model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--device', 1, 'which GPU device to use')
cmd:option('--nsample', 100, 'sample this many words from the language model')
cmd:option('--dumpcsv', false, 'dump training and validation error to CSV file')
cmd:option('--given', '<eos>', 'token to condition generator on')
cmd:text()
local opt = cmd:parse(arg or {})

-- check that saved model exists
assert(paths.filep(opt.xplogpath), opt.xplogpath..' does not exist')

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

local xplog = torch.load(opt.xplogpath)
local lm = xplog.g_net.module

print("Hyper-parameters (xplog.opt):")
print(xplog.opt)

--[[
local trainerr = xplog.trainnceloss or xplog.trainppl
local validerr = xplog.valnceloss or xplog.valppl

print(string.format("Error (epoch=%d): training=%f; validation=%f", xplog.epoch, trainerr[#trainerr], validerr[#validerr]))
--]]
if opt.dumpcsv then
   local csvfile = opt.xplogpath:match('([^/]+)[.]t7$')..'.csv'
   paths.mkdir('learningcurves')
   csvpath = paths.concat('learningcurves', csvfile)
   
   local file = io.open(csvpath, 'w')
   file:write("epoch,trainerr,validerr\n")
   for i=1,#trainerr do
      file:write(string.format('%d,%f,%f\n', i, trainerr[i], validerr[i]))
   end
   file:close()
   
   print("CSV file saved to "..csvpath)
   os.exit()
end

local trainset, validset, testset
if xplog.dataset == 'PennTreeBank' then
   print"Loading Penn Tree Bank test set"
   trainset, validset, testset = dl.loadPTB({50, 1, 1})
   assert(trainset.vocab['the'] == xplog.vocab['the'])
end

print(lm)

lm:forget()
lm:evaluate()

seqgen = lm:findModules('nn.SequenceGenerator')
seqgen[1].ngen = opt.nsample

local sampletext = {}
local prevword = assert(trainset.vocab[opt.given], "Unknown token : "..opt.given)
assert(prevword)
local inputs = torch.LongTensor(1,1) -- seqlen x batchsize
inputs[{1,1}] = prevword
if opt.cuda then inputs = inputs:cuda() end
local output = lm:forward(inputs)

for i=1,output:size(1) do
   local sample = output[i][1]
   local currentword = trainset.ivocab[sample]
   table.insert(sampletext, currentword)
end
print(table.concat(sampletext, ' '))
