import torch
from torch import nn

softmax = nn.Softmax(dim=1)
mse = torch.nn.MSELoss(reduction='none')
bce = torch.nn.BCELoss(reduction='none')


# One level of VAEs
class DTP(nn.Module):


    def __init__(self, num_layer, in_size, hidden_size, out_size):
        super().__init__()

        self.num_layers = num_layer
        self.input_sz = in_size
        self.hidden_sz = hidden_size
        self.out_sz = out_size
        self.encoders, self.decoders = self.create_layers()
        self.e_optims, self.d_optims = self.create_optims()



    def create_layers(self):

        #Add initial layer to encoders and decoders. Add another to encoder since it has one more layer than decoders
        encoders = [nn.Linear(self.input_sz, self.hidden_sz), nn.Linear(self.hidden_sz, self.hidden_sz)]
        decoders = [nn.Linear(self.hidden_sz, self.hidden_sz)]

        #Add remaining layers except output layer
        for n in range(self.num_layers-3):
            encoders.append(nn.Linear(self.hidden_sz, self.hidden_sz))
            decoders.append(nn.Linear(self.hidden_sz, self.hidden_sz))

        #Add output layer
        encoders.append(nn.Linear(self.hidden_sz, self.out_sz))
        decoders.append(nn.Linear(self.out_sz, self.hidden_sz))

        return nn.ModuleList(encoders), nn.ModuleList(decoders)


    def create_optims(self):

        encoder_optims = [torch.optim.Adam(self.encoders[0].parameters())]
        decoder_optims = []

        for n in range(self.num_layers-1):

            encoder_optims.append(torch.optim.Adam(self.encoders[n+1].parameters()))
            decoder_optims.append(torch.optim.Adam(self.decoders[n].parameters()))

        return encoder_optims, decoder_optims


    def compute_values(self, h, x):

        #First h is the input
        h[0] = x

        #Compute all hs except last one. Use Tanh as in paper
        for i in range(1, self.num_layers):
            h[i] = torch.tanh(self.encoders[i-1](h[i-1].detach()))

        #Compute last layer using softmax
        h[-1] = softmax(self.encoders[-1](h[-2].detach()))


    def compute_targets(self, h_hat, h, global_target, p=False):
        with torch.no_grad():
            # Compute global target, h_hat.
            # Here h_hat = h + .5 * dL/dh. We assume MSE is used, so h_hat just equals the target
            h_hat[-1] = global_target

            for i in reversed(range(self.num_layers - 1)):

                #h_hat_m = h_m - g(h_m+1) + g(h_hat_m+1)
                h_hat[i] = h[i+1] - self.decoders[i](h[i+2]) + self.decoders[i](h_hat[i+1])





    def train_decoders(self, h):

        for i in range(self.num_layers - 1):

            # Check
            if h[i+2].grad != None:
                h[i+2].grad.zero_()
            if h[i+1].grad != None:
                h[i+1].grad.zero_()

            # Pass h at level i+2 through decoder to get reconstruction at layer i+1.
            reconstruction = self.decoders[i](h[i+2].detach())

            #Compute loss
            loss = torch.mean(mse(reconstruction, h[i+1].detach()).sum(-1))

            #compute gradients
            self.d_optims[i].zero_grad()
            loss.backward()
            self.d_optims[i].step()






    def train_encoders(self, h, h_hat):

        for i in range(0, self.num_layers):
            #Check


            if h_hat[i].grad != None:
                h_hat[i].grad.zero_()
            if h[i+1].grad != None:
                h[i+1].grad.zero_()

            #Compute loss
            if i < (self.num_layers - 1):
                loss = torch.mean(mse(h[i+1], h_hat[i].detach()).sum(-1))
            else:
                loss = torch.mean(bce(h[i+1], h_hat[i].detach()).sum(-1))

            #Compute gradients
            self.e_optims[i].zero_grad()
            loss.backward()
            self.e_optims[i].step()

        return loss






