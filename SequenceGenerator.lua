------------------------------------------------------------------------
--[[ SequenceGenerator ]]--
-- Encapsulates an rnn. 
-- Input is a sequence of size: seqlen x batchsize x hiddensize
-- Output is a sequence of size: genlen x batchsize x hiddensize
-- Applies the rnn on the input sequence of seqlen time-steps.
-- Then it generates the output sequence. Each generated output is 
-- fed back in as the next output for genlen time-steps.
-- This is also why the input and output both have size hiddensize.
------------------------------------------------------------------------
local SequenceGenerator, parent = torch.class('nn.SequenceGenerator', 'nn.AbstractSequencer')
local _ = require 'moses'

SequenceGenerator.dpnn_mediumEmpty = _.clone(nn.Module.dpnn_mediumEmpty)
table.insert(SequenceGenerator.dpnn_mediumEmpty, '_output')

function SequenceGenerator:__init(rnn, gen, ngen)
   parent.__init(self)
   if not torch.isTypeOf(rnn, 'nn.Module') then
      error"SequenceGenerator: expecting nn.Module instance at arg 1"
   end
   if not torch.isTypeOf(gen, 'nn.Module') then
      error"SequenceGenerator: expecting nn.Module instance at arg 2"
   end
   
   assert(torch.type(ngen) == 'number')
   self.ngen = ngen
   
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   rnn = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   gen = (not torch.isTypeOf(gen, 'nn.AbstractRecurrent')) and nn.Recursor(gen) or gen
   self.modules = {rnn, gen}
   
   self.hidden = {}
   self.output = {}
   self.tableoutput = {}
   self.tablegradInput = {}
   
   -- table of buffers used for evaluation
   self._output = {}
   
   -- default is to forget previous inputs before each forward()
   self._remember = 'neither'
end

function SequenceGenerator:updateOutput(input)
   local seqlen
   if torch.isTensor(input) then
      seqlen = input:size(1) 
   else
      assert(torch.type(input) == 'table', "expecting input table or tensor")
      seqlen = #input
   end

   local rnn, gen = unpack(self.modules)
   rnn:maxBPTTstep(seqlen+self.ngen-1)
   gen:maxBPTTstep(self.ngen)
   
   if self.train ~= false then 
      -- TRAINING
      if not (self._remember == 'train' or self._remember == 'both') then
         rnn:forget()
         gen:forget()
      end
      
      -- condition the generator (forward the input through the module)
      self.hidden = {}
      for step=1,seqlen do
         self.hidden[1] = rnn:updateOutput(input[step])
      end
      
      self.tableoutput = {}
      
      -- generate sequence (forward generated outputs as input, and so on)
      for step=1,self.ngen do
         if step > 1 then
            self.hidden[step] = rnn:updateOutput(self.tableoutput[step-1])
         end
         self.tableoutput[step] = gen:updateOutput(self.hidden[step])
      end
      
      if torch.isTensor(input) then
         self.output = torch.isTensor(self.output) and self.output or self.tableoutput[1].new()
         self.output:resize(self.ngen, unpack(self.tableoutput[1]:size():totable()))
         for step=1,self.ngen do
            self.output[step]:copy(self.tableoutput[step])
         end
      else
         self.output = self.tableoutput
      end
   else 
      -- EVALUATION
      if not (self._remember == 'eval' or self._remember == 'both') then
         rnn:forget()
         gen:forget()
      end
      
      -- condition
      local hidden
      for step=1,seqlen do
         hidden = rnn:updateOutput(input[step])
      end
      
      local output = gen:updateOutput(hidden)
         
      -- during evaluation, recurrent modules reuse memory (i.e. outputs)
      -- so we need to copy each output into our own table or tensor
      if torch.isTensor(input) then
      
         self.output = torch.isTensor(self.output) and self.output or output.new()
         self.output:resize(self.ngen, unpack(output:size():totable()))
         self.output[1]:copy(output)
         
         -- generate
         for step=2,self.ngen do
            hidden = rnn:updateOutput(output)
            output = gen:updateOutput(hidden)
            self.output[step]:copy(output)
         end
      else
         error"Not Implemented"
         self.tableoutput[1] = nn.rnn.recursiveCopy(
            self.tableoutput[1] or table.remove(self._output, 1), 
            self.modules[1]:updateOutput(output)
         )
         
         -- generate
         for step=2,self.ngen do
            self.tableoutput[step] = nn.rnn.recursiveCopy(
               self.tableoutput[step] or table.remove(self._output, 1), 
               self.modules[1]:updateOutput(output)
            )
         end
         
         -- remove extra output tensors (save for later)
         for i=self.ngen+1,#self.tableoutput do
            table.insert(self._output, self.tableoutput[i])
            self.tableoutput[i] = nil
         end
         self.output = self.tableoutput
      end
   end
   
   return self.output
end

function SequenceGenerator:updateGradInput(input, gradOutput)
   local seqlen
   if torch.isTensor(input) then
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      assert(gradOutput:size(1) == self.ngen, "gradOutput should have ngen time-steps")
      seqlen = input:size(1)
   else
      assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
      assert(#gradOutput == self.ngen, "gradOutput should have ngen time-steps")
      seqlen = #input
   end
   
   local rnn, gen = unpack(self.modules)
   
   -- back-propagate through time
   self.gradHidden = {}
   for step=self.ngen,1,-1 do
      self.gradHidden[step] = gen:updateGradInput(self.hidden[step], gradOutput[step])
      if step > 1 then
         rnn:updateGradInput(self.output[step-1], self.gradHidden[step])
      end
   end
   
   self.tablegradinput = {}
   self.tablegradinput[seqlen] = rnn:updateGradInput(input[seqlen], self.gradHidden[1])
   
   if seqlen > 1 then
      self._gradZero = self._gradZero or self.gradHidden[1].new()
      self._gradZero:resizeAs(self.gradHidden[1]):zero()
   end
   
   for step=seqlen-1,1,-1 do
      self.tablegradinput[step] = rnn:updateGradInput(input[step], self._gradZero)
   end
   
   if torch.isTensor(input) then
      self.gradInput = torch.isTensor(self.gradInput) and self.gradInput or self.tablegradinput[1].new()
      self.gradInput:resize(seqlen, unpack(self.tablegradinput[1]:size():totable()))
      for step=1,seqlen do
         self.gradInput[step]:copy(self.tablegradinput[step])
      end
   else
      self.gradInput = self.tablegradinput
   end

   return self.gradInput
end

function SequenceGenerator:accGradParameters(input, gradOutput, scale)
   local seqlen
   if torch.isTensor(input) then
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      assert(gradOutput:size(1) == self.ngen, "gradOutput should have ngen time-steps")
      seqlen = input:size(1)
   else
      assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
      assert(#gradOutput == self.ngen, "gradOutput should have ngen time-steps")
      seqlen = #input
   end
   
   local rnn, gen = unpack(self.modules)
   
   -- back-propagate through time
   for step=self.ngen,1,-1 do
      gen:accGradParameters(self.hidden[step], gradOutput[step], scale)
      if step > 1 then
         rnn:accGradParameters(self.output[step-1], self.gradHidden[step], scale)
      end
   end
   
   rnn:accGradParameters(input[seqlen], self.gradHidden[1], scale)
   
   for step=seqlen-1,1,-1 do
      rnn:accGradParameters(input[step], self._gradZero, scale)
   end
   
end

function SequenceGenerator:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   error"Not Implemented"  
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
-- Essentially, forget() isn't called on rnn module when remember is on
function SequenceGenerator:remember(remember)
   assert(remember and remember == 'neither', "Only supports 'neither' for now")
   self._remember = (remember == nil) and 'both' or remember
   local _ = require 'moses'
   assert(_.contains({'both','eval','train','neither'}, self._remember), 
      "SequenceGenerator : unrecognized value for remember : "..self._remember)
   return self
end

function SequenceGenerator:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
      -- empty temporary output table
      self._output = {}
      -- empty output table (tensor mem was managed by seq)
      self.tableoutput = nil
      self.hidden = nil
      self.gradHidden = nil
   end
   parent.training(self)
end

function SequenceGenerator:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
      -- empty output table (tensor mem was managed by rnn)
      self.tableoutput = {}
      self.hidden = nil
      self.gradHidden = nil
   end
   parent.evaluate(self)
   assert(self.train == false)
end

function SequenceGenerator:clearState()
   if torch.isTensor(self.output) then
      self.output:set()
      self.gradInput:set()
   else
      self.output = {}
      self.gradInput = {}
   end
   self._output = {}
   self.tableoutput = {}
   self.hidden = {}
   self.gradHidden = {}
   self.tablegradinput = {}
   self.modules[1]:clearState()
   self.modules[2]:clearState()
end

function SequenceGenerator:__tostring__()
   local tab = '  '
   local line = '\n'
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {'
   str = str .. line .. tab .. 'rnn : ' .. tostring(self.modules[1]):gsub(line, line .. tab .. ext)
   str = str .. line .. tab .. 'gen     : ' .. tostring(self.modules[2]):gsub(line, line .. tab .. ext)
   str = str .. line .. '}'
   return str
end
