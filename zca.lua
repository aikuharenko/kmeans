function gen_data(n)

	x = torch.Tensor(2,n)
	x[1] = x[1]:normal(0,1)
	x[2] = x[2]:normal(0, 3)
	local ang=3.14 / 8
	local rotm = torch.Tensor({{math.cos(ang), -math.sin(ang)}, {math.sin(ang), math.cos(ang)}})
	x = rotm * x

	return x

end

function whiten_images(ims, epsilon)

	local t = sys.clock()

	local m = (#ims)[1]
	local sx = (#ims)[2]
	local n = sx * sx

	local x = nn.Reshape(m, n):forward(ims)

	for i = 1, n do
		x[i] = x[i] / 256
		avg = x[i]:mean()
		x[i] = x[i] - avg
	end

	x = x:t()

	local sigma = x * x:t() / m
	local U, S, V = torch.svd(sigma)

	S = S + epsilon
	S:sqrt()
	local ones = torch.Tensor(n):fill(1)
	ones:cdiv(S)
	ones = ones:diag()
	local SS = ones
	x = U * SS * U:t() * x	
	
	x = x:t()

	print(sys.clock() - t .. ' seconds to whiten images')

	return x, U, SS

end


function zca_patch(patch, U, S)

	local sx = (#patch)[2]
	p = patch:reshape(sx * sx)
	p = U * S * U:t() * p	
	p = p / p:norm()

	return p

end

--[[function zca_convolve(image, D, U, S)

	local sx = (#D)[2]
	local m = (#D)[1]
	dim = (#image)[1]
	sx0 = dim - sx + 1
	
	conv = torch.Tensor(m, sx0, sx0):fill(1)

	for y = 1, sx0 do 
		for x = 1, sx0 do

			p = image[{{y, y + sx - 1}, {x, x + sx - 1}}]:clone()
			p = zca_patch(p)
			r = D * p
			
		end
	end

	return conv

end--]]

--[[x = gen_data(100)
gnuplot.figure(1)
gnuplot.plot(x[1], x[2], '+')

x2 = zca_whitening(x)

gnuplot.figure(2)
gnuplot.plot(x2[1], x2[2], '+')
--]]
