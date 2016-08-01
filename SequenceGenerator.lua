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

function SequenceGenerator:__init(rnn, ngen)
   parent.__init(self)
   if not torch.isTypeOf(rnn, 'nn.Module') then
      error"SequenceGenerator: expecting nn.Module instance at arg 1"
   end
   
   assert(torch.type(ngen) == 'number')
   self.ngen = ngen
   
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.module = (not torch.isTypeOf(rnn, 'nn.AbstractRecurrent')) and nn.Recursor(rnn) or rnn
   self.modules = {self.module}
   
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

   self.module:maxBPTTstep(seqlen+self.ngen-1)
   
   if self.train ~= false then 
      -- TRAINING
      if not (self._remember == 'train' or self._remember == 'both') then
         self.module:forget()
      end
      
      -- condition the generator (forward the input through the module)
      self.tableoutput = {}
      for step=1,seqlen do
         self.tableoutput[1] = self.module:updateOutput(input[step])
      end
      
      -- generate sequence (forward generated outputs as input, and so on)
      for step=2,self.ngen do
         self.tableoutput[step] = self.module:updateOutput(self.tableoutput[step-1])
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
         self.module:forget()
      end
      
      -- condition
      local output
      for step=1,seqlen do
         output = self.module:updateOutput(input[step])
      end
         
      -- during evaluation, recurrent modules reuse memory (i.e. outputs)
      -- so we need to copy each output into our own table or tensor
      if torch.isTensor(input) then
      
         self.output = torch.isTensor(self.output) and self.output or output.new()
         self.output:resize(self.ngen, unpack(output:size():totable()))
         self.output[1]:copy(output)
         
         -- generate
         for step=2,self.ngen do
            output = self.module:updateOutput(output)
            self.output[step]:copy(output)
         end
      else
      
         self.tableoutput[1] = nn.rnn.recursiveCopy(
            self.tableoutput[1] or table.remove(self._output, 1), 
            self.module:updateOutput(output)
         )
         
         -- generate
         for step=2,self.ngen do
            self.tableoutput[step] = nn.rnn.recursiveCopy(
               self.tableoutput[step] or table.remove(self._output, 1), 
               self.module:updateOutput(output)
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
   
   -- back-propagate through time
   for step=self.ngen,2,-1 do
      self.module:updateGradInput(self.output[step-1], gradOutput[step])
   end
   
   self.tablegradinput = {}
   self.tablegradinput[seqlen] = self.module:updateGradInput(input[seqlen], gradOutput[1])
   
   if seqlen > 1 then
      self._gradZero = self._gradZero or gradOutput.new()
      self._gradZero:resizeAs(gradOutput[1]):zero()
   end
   
   for step=seqlen-1,1,-1 do
      self.tablegradinput[step] = self.module:updateGradInput(input[step], self._gradZero)
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
   
   -- back-propagate through time 
   for step=self.ngen,2,-1 do
      self.module:accGradParameters(self.output[step-1], gradOutput[step], scale)
   end   
   
   self.module:accGradParameters(input[seqlen], gradOutput[1], scale)
   
   for step=seqlen-1,1,-1 do
      self.module:accGradParameters(input[step], self._gradZero, scale)
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
   end
   parent.training(self)
end

function SequenceGenerator:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
      -- empty output table (tensor mem was managed by rnn)
      self.tableoutput = {}
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
   self.tablegradinput = {}
   self.module:clearState()
end

SequenceGenerator.__tostring__ = nn.Decorator.__tostring__
