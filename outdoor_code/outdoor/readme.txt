self.fc3 = nn.Linear(2048, 2)
有的模型是在4的条件下生成的，如果报错，可以改成4试试（loss去掉camera参数后，网络输出没改）