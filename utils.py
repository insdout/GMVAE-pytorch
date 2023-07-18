from models import GMVAE, GMVAE2
from loss import TotalLoss, MSE, BCELogits, Loss2
from modules import Qy_x, Qz_xy, Px_z
from modules import EncoderFC, DecoderFC, EncoderLSTM, DecoderLSTM
import json
import numpy as np
import torch


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def get_model(k, encoder_type, input_size, hidden_size, latent_dim,
              recon_loss_type="MSE", return_probs=False, eps=0,
              encoder_kwargs={}, decoder_kwargs={}, model_name="GMVAE2", loss_name="Loss"):

    if model_name == "GMVAE":
        if encoder_type == "FC":
            encoder = EncoderFC(input_size=input_size, hidden_size=hidden_size)
            decoder = DecoderFC(input_size=input_size, hidden_size=hidden_size, latent_dim=latent_dim, return_probs=return_probs)
        elif encoder_type == "LSTM":
            encoder = EncoderLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                **encoder_kwargs
                )
            decoder = DecoderLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                latent_dim=latent_dim,
                **decoder_kwargs
                )
        else:
            raise ValueError(f"Encoder {encoder_type} is not implemented.")

        Qy_x_net = Qy_x(encoder=encoder, k=k, enc_out_dim=hidden_size)
        Qz_xy_net = Qz_xy(encoder=encoder, enc_out_dim=hidden_size, k=k, hidden_size=hidden_size, latent_dim=latent_dim)
        Px_z_net = Px_z(decoder=decoder, k=k)
        model = GMVAE(k=k, Qy_x_net=Qy_x_net, Qz_xy_net=Qz_xy_net, Px_z_net=Px_z_net)
        model.apply(init_weights)
        print(f"Model {model_name} created.")
    
    elif model_name == "GMVAE2":
        model = GMVAE2(input_size, k, latent_dim, hidden_size)
        model.apply(init_weights)
        print(f"Model {model_name} created.")
    else:
        raise ValueError(f"Model {model_name} is not implemented.")

    if loss_name == "Loss":
        if recon_loss_type == "MSE":
            recon_loss = MSE()
        else:
            recon_loss = BCELogits(eps=eps)
        loss = TotalLoss(k, recon_loss)
        print(f"Loss {loss_name} created.")
    elif loss_name == "Loss2":
        loss = Loss2()
        print(f"Loss {loss_name} created.")
    else:
        raise ValueError(f"Loss {loss_name} is not implemented.")

    return model, loss


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

