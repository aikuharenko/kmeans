require 'torch'
require 'paths'
require 'sys'
require 'image'
require 'nn'

function gen_data_for_kmeans(m, n, k)

	local centers = torch.Tensor(k, n):uniform() * 5
	local points = torch.Tensor(m, n):uniform()
	local labels = torch.Tensor(m):fill(1)

	local mpart = math.floor(m / k)
	for i = 1, k do

		labels[{{(i-1) * mpart + 1 , i * mpart}}]:fill(i)

	end
	return points, labels

end

function show_kmeans(points, labels, centers, nfig)

	local m = (#labels)[1]
	local k = (#centers)[1]
	local card = torch.Tensor(k)

	for i = 1, k do
		card[i] = torch.eq(labels, i):sum()	
	end

	lp = {}
	for i = 1, k do
		table.insert(lp, torch.Tensor(card[i], n))
	end

	local idxs = torch.Tensor(k):zero()

	for i = 1, m do

		local j = labels[i]
		idxs[j] = idxs[j] + 1
		lp[j][idxs[j]] = points[i]
	
	end

	for i = 1, k do
		lp[i] = lp[i]:t()
	end

	plt = {}
	for i = 1, k do
		table.insert(plt, {lp[i][1], lp[i][2], '+'})
		table.insert(plt, {torch.Tensor(1):fill(centers[i][1]), torch.Tensor(1):fill(centers[i][2]), '+'})
	end

	gnuplot.figure(nfig)
	gnuplot.plot(plt)

end
------------------------------------------------------------------------------
function init_dictionary(points, k)

	local m = (#points)[1]
	local n = (#points)[2]

	local D = torch.Tensor(k, n)
	
	for i = 1, k do
		local idx = torch.uniform() * (m - 1) + 1
		D[i] = points[idx]
	end

	local normD = torch.norm(D, 2, 2)
	normD = torch.expand(normD, k, n)
	D:cdiv(normD)

	return D

end

function assign_points_slow(points, centers)

	local m = (#points)[1]
	local k = (#centers)[1]
	local err = 0

	labels = torch.Tensor(m)

	for i = 1, m do

		local dist = 1e+5
		local pos = 1

		for j = 1, k do

			local d = points[i]:dist(centers[j])
			if d < dist then
				dist = d
				pos = j
			end
		end
					
		labels[i] = pos
		err = err + dist

	end			
	
	return labels, err

end

function assign_points(points, centers)

	local m = (#points)[1]
	local n = (#points)[2]
	local k = (#centers)[1]
	local errs = 0

	local labels = torch.Tensor(m)
	local dists = torch.Tensor(k, m)

	for i = 1, k do
		
		ci = torch.Tensor(1, n)
		ci[1] = centers[i]

		diff = torch.expand(ci, m, n)
		diff = diff - points
		dists[i] = torch.norm(diff, 2, 2)[{{},{1}}]		

	end			
	
	errs2, labels = torch.min(dists, 1)
	local err = torch.sum(errs2)
	labels = labels[1]

	return labels, err

end

function compute_D(points, labels, k)

	local m = (#points)[1]
	local n = (#points)[2]
	
	local D = torch.Tensor(k, n):zero()

	for i = 1, m do
		
		local j = labels[i]
		D[j] = D[j] + points[i]

	end

	local normD = torch.norm(D, 2, 2)
	normD = torch.expand(normD, k, n)
	D:cdiv(normD)

	return D

end

function train_kmeans(points, k, maxiters)
--m = 10000
--points, labels = gen_data_for_kmeans(m, n, k)
--centers = init_centers(points, k)
--show_kmeans(points, labels, centers, 1)

	D = init_dictionary(points, k)	
	err = torch.Tensor(maxiters)

	for i = 1, maxiters do
		
		local t = sys.clock()
		labels, err[i] = assign_points(points, D)
		D = compute_D(points, labels, k)
	
		print('iter ' .. i .. ' error ' .. err[i] ..' time ' .. sys.clock() - t)

	end	

	return D, labels

end




