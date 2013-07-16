function load_dataset()
--
	if not paths.dirp('cifar-10-batches-t7') then
	   local www = 'http://data.neuflow.org/data/cifar-10-torch.tar.gz'
	   local tar = sys.basename(www)
	   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
	end

	local t1 = sys.clock()
	
	train_images = torch.Tensor(trsize, 1, 32, 32):float()
	test_images = torch.Tensor(tesize, 1, 32, 32):float()
	train_labels = torch.Tensor(trsize):float()
	test_labels = torch.Tensor(tesize):float()

	--load train images and labels
	for i = 0,4 do

		subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
		subset.data = subset.data:t()
	
		for j = 1, 10000 do
		
			local im = subset.data[j]
			im = im:reshape(3, 32, 32):float()		
			train_images[10000 * i + j] = image.rgb2y(im)			

		end

		train_labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels

	end

	train_labels = train_labels + 1

	--load test images and labels
	local subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
	subset.data = subset.data:t()

	for j = 1, 10000 do
	
		local im = subset.data[j]
		im = im:reshape(3, 32, 32):float()		
		test_images[j] = image.rgb2y(im)			

	end

	test_labels[{{1, 10000} }] = subset.labels
	test_labels = test_labels + 1


	train_images = train_images:float()
	train_labels = train_labels:float()
	test_images = test_images:float()
	test_labels = test_labels:float()

	print(sys.clock() - t1 .. ' seconds to load dataset')

	return train_images, train_labels, test_images, test_labels

end

function normilize_data(ims)
	
	local t = sys.clock()

	local neighborhood = image.gaussian1D(7)
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

	local m = (#ims)[1]
	
	for i = 1, m do

		ims[i] = normalization:forward(ims[i])

	end

	print(sys.clock() - t .. ' seconds to normalize ' .. m .. ' images')

	return ims
end

