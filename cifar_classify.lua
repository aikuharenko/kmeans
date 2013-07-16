require 'optim'

dofile('prepare_cifar.lua')
dofile('zca.lua')
dofile('kmeans.lua')
dofile('train_softmax.lua')

function extract_patches(images, m, s)

	local t = sys.clock()

	local n = (#images)[1]
	local rgen = torch.rand(m, 3)
	local patches = torch.Tensor(m, s, s)

	for i = 1, m do
		
		local idx = math.floor(rgen[i][1] * (n - 1)) + 1			
		local y = math.floor(rgen[i][2] * (32 - s)) + 1
		local x = math.floor(rgen[i][2] * (32 - s)) + 1
		patches[i] = images[{{idx}, {1}, {y, y + s - 1}, {x, x + s - 1}}]

	end

	print(sys.clock() - t .. ' seconds to extract ' .. m .. ' patches')

	return patches

end

function do_kmeans(train_k_means)

	local fname = 'filters/kmeans_centers_k' .. k .. '_m' .. m .. '_sx' .. sx .. '.asc'

	if train_k_means then

		patches2 = extract_patches(train_images, m, sx)	
		patches, U, S = whiten_images(patches2, 1e-1)
	
		--compute kmeans dictionary
		D, temp = train_kmeans(patches, k, niters)

		--save dictionary to file
		local zca = {
			D = D,
			U = U,
			S = S
		}
		torch.save(fname, obj)

	else

		local fname = 'filters/kmeans_centers_k' .. k .. '_m' .. m .. '_sx' .. sx .. '.asc'
		zca = torch.load(fname)		

	end

	zca.D = zca.D:float()
	zca.U = zca.U:float()
	zca.S = zca.S:float()

end

function f(img)

	local features = zca_convolve(img, D, U, S)
	return img:reshape(fdim)

end

function zca_convolve(im, zca)

	local n = (#D)[2]
	local sx = math.sqrt(n)
	local m = (#D)[1]
	dim = (#im)[2]
	sx0 = dim - sx + 1

	conv = torch.Tensor(m, sx0, sx0)

	for y = 1, sx0 do 
		for x = 1, sx0 do

			p = im[{{y, y + sx - 1}, {x, x + sx - 1}}]:clone()
			p = zca_patch(p, U, S)
			r = D * p
			conv[{{},{y},{x}}] = r			

		end
	end	

	return conv
	
end

trsize = 50000
tesize = 10000

sx = 8
k = 100
n = sx * sx
m = 50000
niters = 10
fdim = 100*25*25

opt = {}
opt.train_kmeans = false

opt.batch_size = 10
opt.learning_rate = 1e-6
opt.learning_rate_decay = 1e-8
opt.nepochs = 10
opt.cuda = false

--train_images, train_labels, test_images, test_labels = load_dataset()
--train_images = normilize_data(train_images)
--test_images = normilize_data(test_images)

do_kmeans(opt.train_kmeans)
--image.display({image=D:reshape(k, sx, sx), padding=2, zoom=2, nrow=10})

--t = zca_convolve(train_images[1][1], D, U, S)


train_softmax(train_images, train_labels, test_images, test_labels, f, fdim, opt)

function get_convolve_model()

	model = nn.Sequential()
	model:add(nn.SpatialConvolution(1, 100, sx, sx))
	weights = model:get(1).weight
	weights[{{1,100},{1},{},{}}] = D
	--weights = weights:float()
	model:float()

end


















