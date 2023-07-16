from models import GMVAE
from loss import TotalLoss, MSE, BCELogits
from modules import Qy_x, Qz_xy, Px_z
from modules import EncoderFC, DecoderFC, EncoderLSTM, DecoderLSTM


def get_model(k, encoder_type, input_size, hidden_size, latent_dim,
              recon_loss_type="MSE", return_probs=False, eps=0,
              encoder_kwargs={}, decoder_kwargs={}):

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
        raise ValueError(f"{encoder_type} is not implemented.")

    Qy_x_net = Qy_x(encoder=encoder, k=k, enc_out_dim=hidden_size)
    Qz_xy_net = Qz_xy(encoder=encoder, enc_out_dim=hidden_size, k=k, hidden_size=hidden_size, latent_dim=latent_dim)
    Px_z_net = Px_z(decoder=decoder, k=k)
    model = GMVAE(k=k, Qy_x_net=Qy_x_net, Qz_xy_net=Qz_xy_net, Px_z_net=Px_z_net)

    if recon_loss_type == "MSE":
        recon_loss = MSE()
    else:
        recon_loss = BCELogits(eps=eps)
    loss = TotalLoss(k, recon_loss)
    return model, loss
