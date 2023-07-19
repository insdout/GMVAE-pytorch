from models import GMVAE, GMVAE2
from loss import TotalLoss, MSE, BCELogits, Loss2
from modules import Qy_x, Qz_xy, Px_z
from modules import EncoderFC, DecoderFC, EncoderLSTM, DecoderLSTM
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import os
import logging


log = logging.getLogger(__name__)


def init_weights(m):
    """_summary_

    Args:
        m (_type_): _description_
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def get_model(k, encoder_type, input_size, hidden_size, latent_dim,
              recon_loss_type="MSE", return_probs=False, eps=0,
              encoder_kwargs={}, decoder_kwargs={}, model_name="GMVAE2", loss_name="Loss"):
    """_summary_

    Args:
        k (_type_): _description_
        encoder_type (_type_): _description_
        input_size (_type_): _description_
        hidden_size (_type_): _description_
        latent_dim (_type_): _description_
        recon_loss_type (str, optional): _description_. Defaults to "MSE".
        return_probs (bool, optional): _description_. Defaults to False.
        eps (int, optional): _description_. Defaults to 0.
        encoder_kwargs (dict, optional): _description_. Defaults to {}.
        decoder_kwargs (dict, optional): _description_. Defaults to {}.
        model_name (str, optional): _description_. Defaults to "GMVAE2".
        loss_name (str, optional): _description_. Defaults to "Loss".

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
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
        log.info(f"Model {model_name} created.")

    elif model_name == "GMVAE2":
        model = GMVAE2(input_size, k, latent_dim, hidden_size)
        model.apply(init_weights)
        log.info(f"Model {model_name} created.")
    else:
        raise ValueError(f"Model {model_name} is not implemented.")

    if loss_name == "Loss":
        if recon_loss_type == "MSE":
            recon_loss = MSE()
        else:
            recon_loss = BCELogits(eps=eps)
        loss = TotalLoss(k, recon_loss)
        log.info(f"Loss {loss_name} created.")
    elif loss_name == "Loss2":
        loss = Loss2()
        log.info(f"Loss {loss_name} created.")
    else:
        raise ValueError(f"Loss {loss_name} is not implemented.")

    return model, loss


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def plot_id_history(history, output_dir):
    """_summary_

    Args:
        history (_type_): _description_
        output_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    keys = list(history.keys())
    history_len = len(history[keys[0]]["x_hat"])
    log.info(f"Length of ids_history: {history_len}")
    num_rows = len(keys)
    num_columns = 3

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, sharex=False,
                            sharey=False, figsize=(8, 10),
                            gridspec_kw={'width_ratios': [3, 1, 1]})

    def animate_diff(i, data):
        ids = keys
        plots = []
        for row in range(num_rows):
            for col in range(num_columns):
                row_id = ids[row]
                axs[row, col].clear()
            axs[row, 0].set_xticks([])
            axs[row, 1].set_xticks([])
            axs[row, 1].set_yticks([])
            axs[row, 2].set_xticks([])
            axs[row, 2].set_yticks([])
            axs[row, 0].set_ylim([0, 1])
            axs[row, 0].set_yticks(range(0, 2))
            if row == 0:
                axs[row, 0].set_title("P(y|x)")
                axs[row, 1].set_title("True X")
                axs[row, 2].set_title("Reconstructed X")
            if row == num_rows - 1:
                axs[row, 0].set_xticks(range(0, 10))
                axs[row, 0].set_xlabel("y")
            # axs[row, 0].set_ylabel("p")
            plots.append(axs[row, 0].bar(range(10), history[row_id]["qy"][i]))
            plots.append(axs[row, 1].imshow(np.reshape(history[row_id]["x_true"], (28, 28))))
            plots.append(axs[row, 2].imshow(np.reshape(history[row_id]["x_hat"][i], (28, 28))))
        plt.suptitle(f"Epoch: {i+1}")
        return plots

    ani = FuncAnimation(fig, animate_diff, fargs=[history],  interval=250,
                        blit=False, repeat=True, frames=history_len)
    
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    gif_path = os.path.join(img_dir, "gen_over_epochs.gif")
    ani.save(gif_path, dpi=100, writer=PillowWriter(fps=5))
    log.info('saved image at ' + gif_path)
    plt.close('all')


def plot_training_curves(history, output_dir):
    """_summary_

    Args:
        history (_type_): _description_
        output_dir (_type_): _description_
    """

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    ax[0].plot(history['train_cond_entropy'], label='Train H(y|x)')
    ax[0].plot(history['test_cond_entropy'], label='Test H(y|x)')
    ax[0].set_xlabel("Epoch")
    ax[0].set_title("H(y|x)")

    ax[1].plot(history['train_loss'], label='Train Loss')
    ax[1].plot(history['test_loss'], label='Test Loss')
    ax[1].set_xlabel("Epoch")
    ax[1].set_title("Loss")

    ax[2].plot(history['train_accuracy'], label='Test Accuracy')
    ax[2].plot(history['test_accuracy'], label='Test Accuracy')
    ax[2].set_xlabel("Epoch")
    ax[2].set_title("Accuracy")

    ax[0].legend(loc='upper left')
    ax[1].legend(loc='upper left')
    ax[2].legend(loc='upper left')

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, "training_curves.png")
    plt.savefig(img_path)
    log.info('saved image at ' + img_path)
    plt.close('all')
