require 'nngraph'

--Using csv2tensor (install with luarocks)
--csv2tensor = require 'csv2tensor'
--data, column_names = csv2tensor.load("./quiebras-spain-2005.csv") 

-- Read data from CSV to tensor
local csvFile = io.open('./quiebras-spain-2005.csv', 'r')  
local header = csvFile:read()
local data = torch.Tensor(2860,36) --2860,36--
local i = 0  
for line in csvFile:lines('*l') do  
  i = i + 1
  print(i)
  local l = line:split(',')
  for key, val in ipairs(l) do
  	if val=="#NULL!" then
  		val=0 --what to do with NULL?--
  	end
  	print(val)
    data[i][key] = val
  end
end

csvFile:close() 


print(data)

-- it is common style to mark inputs with identity nodes for clarity.
--input = nn.Identity()()
input = data
-- each hidden layer is achieved by connecting the previous one
-- here we define a single hidden layer network
h1 = nn.Tanh()(nn.Linear(20, 10)(input))
output = nn.Linear(10, 1)(h1)
mlp = nn.gModule({input}, {output})

x = torch.rand(20)
dx = torch.rand(1)
mlp:updateOutput(x)
mlp:updateGradInput(x, dx)
mlp:accGradParameters(x, dx)

-- draw graph (the forward graph, '.fg')
-- this will produce an SVG in the runtime directory
--graph.dot(mlp.fg, 'MLP', 'MLP')
--itorch.image('MLP.svg')
