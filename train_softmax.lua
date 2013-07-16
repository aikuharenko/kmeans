require 'cutorch'
require 'cunn'

function train_softmax(train_images, train_labels, test_images, test_labels, extract_features, n, opt)

	local epoch = 1
	local time = sys.clock()
	local trsize = (#train_images)[1]

	classes = {'1','2','3','4','5','6','7','8','9','0'}
	train_confusion = optim.ConfusionMatrix(classes)
	test_confusion = optim.ConfusionMatrix(classes)
	Logger = optim.Logger('train.log')
	ce_logger = optim.Logger('ce_train.log')

	model, criterion = get_classifier(n, opt)
	parameters, gradParameters = model:getParameters()

	optimState = {
		learningRate = opt.learning_rate,
		weightDecay = 0,
		momentum = 0,
		learningRateDecay = opt.learning_rate_decay
	}

	for e = 1, opt.nepochs do

		--TRAIN--
		shuffle = torch.randperm(trsize)
		print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batch_size .. ']')
		local time = sys.clock()
		ce_train_err = 0

		for t = 1, trsize, opt.batch_size do

			-- create mini batch
			inputs = {}
			targets = {}

			inputs = torch.Tensor(opt.batch_size, n)
			targets = torch.Tensor(opt.batch_size)
			local j = 0

			for i = t, math.min(t + opt.batch_size - 1, trsize) do
				
				j = j + 1
				local sample = extract_features(train_images[shuffle[i]])
				local input = sample:clone()
				inputs[j] = input
				targets[j] = train_labels[shuffle[i]]

			end

			if opt.cuda then
				inputs = inputs:cuda()			
			else
				inputs = inputs:float()
			end

			-- create closure to evaluate f(X) and df/dX
			local feval = function(x)

				if x ~= parameters then
					parameters:copy(x)
				end

				gradParameters:zero()
				
				outputs = model:forward(inputs)
				f = criterion:forward(outputs, targets)
				df_do = criterion:backward(outputs, targets)
				model:backward(inputs, df_do)				

				for i = 1,opt.batch_size do
					train_confusion:add(outputs[i], targets[i])
				end

				return f, gradParameters

			end
			
			xxx, ce = optim.sgd(feval, parameters, optimState)
			ce_train_err = ce_train_err + ce[1]

		end
		
		ce_train_err = ce_train_err / (trsize / opt.batch_size) 
		local train_time = sys.clock() - time
	
		--END TRAIN--

		--TEST--

		local time = sys.clock()
		print('==> testing on test set:')

		local ce_test_err = 0
		for t = 1, tesize do

			local sample = extract_features(test_images[t])
			input = sample:clone()
			
			if opt.cuda then
				input = input:cuda()
			else
				input = input:float()
			end

			local target = test_labels[t]
			local pred = model:forward(input)
			local ce_err = criterion:forward(pred, target)
			ce_test_err = ce_test_err + ce_err

			test_confusion:add(pred, target)

		end

		ce_test_err = ce_test_err / tesize
		local test_time = sys.clock() - time

		--END TEST

		-- update logger/plot
		train_confusion:updateValids()
		test_confusion:updateValids()
		
		Logger:add{['% train accuracy'] = train_confusion.totalValid * 100,
					['% test accuracy'] = test_confusion.totalValid * 100}
		Logger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
		Logger:plot()

		ce_logger:add{['ce train error'] = ce_train_err,
					['ce test error'] = ce_test_err}
		ce_logger:style{['ce train error'] = '-', ['ce test error'] = '-'}
		ce_logger:plot()

		print('train accuracy=' .. train_confusion.totalValid * 100 .. '. ce error=' .. ce_train_err .. '. time=' .. train_time)
		print('test accuracy=' .. test_confusion.totalValid * 100 .. '. ce error=' .. ce_test_err .. '. time=' .. test_time)
		
		train_confusion:zero()
		test_confusion:zero()
		
	end
	
end

function get_classifier(n, opt)

	local model = nn.Sequential()
	model:add(nn.Linear(n, 10))
	model:add(nn.LogSoftMax())
	model:add(nn.Reshape(10))
	local criterion = nn.ClassNLLCriterion()
	
	if opt.cuda then 	
		model:cuda()
		criterion:cuda()
	else
		model:float()
		criterion:float()
	end

	return model, criterion

end

