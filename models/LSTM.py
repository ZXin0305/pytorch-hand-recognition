import torch
import torch.nn as nn
from IPython import embed


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, lstm_layers, fc_layers, num_classes, t_step=54, num_coordi=45):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers,batch_first=True)
        self.fc = nn.Linear(t_step * num_coordi, t_step * num_coordi)
        self.lstm_layers = lstm_layers
        self.fc_layers = fc_layers

        self.lstm_block = list()
        self.fc_block = list()

        for i in range(lstm_layers):
            self.lstm_block.append(self.lstm)
            setattr(self, 'lstm_%d' %  i, self.lstm_block[i])  
        

        for i in range(fc_layers):
            self.fc_block.append(self.fc)
            # setattr(self, 'fc_%d' %  i, self.fc_block[i])
        self.fc_sequence = nn.Sequential(*self.fc_block)  #一个序列

        self.fc_output_1 = nn.Linear(t_step * num_coordi, num_classes)
        # self.fc_output_2 = nn.Linear(1000, 200)
        # self.fc_output_3 = nn.Linear(200, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        out = None
        # print(x.shape)
        B,H,W = x.shape
        if self.fc_layers != 0:
            x = x.reshape(B,-1)
            out = self.fc_sequence(x)
            out = self.tanh(out)
            out = out.reshape(B,H,W)  # (B, 54, 45)
        else:
            out = x
        hn, cn = None, None
        
        for i in range(self.lstm_layers):
            out, (hn, cn) = eval('self.lstm_'+str(i))(out)

        out = self.tanh(out)
        
        out = out.reshape(B,-1)
        final = self.fc_output_1(out)
        # out = self.fc_output_2(out)
        # final = self.fc_output_3(out)
        return final

if __name__ == "__main__":
    lstm_model = LSTM_model(42,42,6,2, 0,11,1,42)
    input = torch.randn(1,1,42)  # --> batch-size, T, hidden_size 
    final = lstm_model(input)
