require "json"

--[[
En este código vamos a adaptar los datos recibidos por el fichero texto.json
preparándolo para su futuro uso
]]--


local dic = json.load("texto.json");

dataset={};

for i=1,dic["tam"] do
	local input= torch.Tensor(4)
	for j=1,4,1 do
		input[j] = dic["raw_inputs"][i][j];
	end
	local output= torch.Tensor(1)
	output[1]=dic["raw_targets"][i];
	dataset[i] = {input,output};
end

function dataset:size()
  return dic["tam"]
end
