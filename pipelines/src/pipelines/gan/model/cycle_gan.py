import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pipelines.gan.buffer.replay_buffer import ReplayBuffer
from pipelines.gan.config import TrainingGANConfig
from pipelines.gan.model.discriminator import WaveGANDiscriminator
from pipelines.gan.model.generator import CyclicWaveGanGenerator
from pipelines.gan.model.utils import (
    gradients_status,
    update_optimizer_lr,
    weights_init,
)
from pipelines.gan.utils import save_samples


class CycleGan(object):
    """
    A CycleGAN implementation for cyclic audio signal transformation using two discriminators and one generator.
    Supports training, validation, and saving/loading checkpoints.

    Attributes:
        config (TrainingGANConfig): Configuration object containing training parameters.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        validate (bool): Whether to enable validation during training.
        writer (SummaryWriter, optional): TensorBoard writer for logging.
        device (torch.device): Computation device, either 'cuda' or 'cpu'.
        val_g_cost (list): List of generator losses on the validation set.
        train_g_cost (list): List of generator losses on the training set.
        cyclic_loss (list): List of cyclic consistency losses.
        valid_reconstruction (list): List of reconstruction losses during validation.
        discriminator_loss (list): List of discriminator losses.
        generator (CyclicWaveGanGenerator): The generator model.
        discriminator_1 (WaveGANDiscriminator): First discriminator model.
        discriminator_2 (WaveGANDiscriminator): Second discriminator model.
        optimizer_d_1 (optim.Optimizer): Optimizer for the first discriminator.
        optimizer_d_2 (optim.Optimizer): Optimizer for the second discriminator.
        optimizer_g (optim.Optimizer): Optimizer for the generator.
    """

    def __init__(
            self,
            config: TrainingGANConfig,
            train_loader: DataLoader,
            val_loader: DataLoader,
            validate: bool = True,
            writer: Optional[SummaryWriter] = None):
        """
        Initializes the CycleGAN model with specified configurations, models, and optimizers.

        Args:
            config (TrainingGANConfig): Training configuration object.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            validate (bool, optional): Whether to perform validation during training. Defaults to True.
            writer (SummaryWriter, optional): TensorBoard writer for logging. Defaults to None.
        """
        self.config = config
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.validate = validate
        self.writer = writer

        self.val_g_cost = []
        self.train_g_cost = []
        self.cyclic_loss = []
        self.valid_reconstruction = []

        self.discriminator_loss = []

        self.generator = CyclicWaveGanGenerator(
            slice_len=self.config.window_length,
            model_size=self.config.model_capacity_size).to(
            self.device)
        self.generator.apply(weights_init)

        self.discriminator_1 = WaveGANDiscriminator(
            slice_len=config.window_length,
            model_size=config.model_capacity_size).to(
            self.device)
        self.discriminator_1.apply(weights_init)

        self.discriminator_2 = WaveGANDiscriminator(
            slice_len=config.window_length,
            model_size=config.model_capacity_size).to(
            self.device)
        self.discriminator_2.apply(weights_init)

        self.optimizer_d_1 = optim.Adam(
            self.discriminator_1.parameters(), lr=config.lr_d, betas=(
                config.beta1, config.beta2))
        self.optimizer_d_2 = optim.Adam(
            self.discriminator_2.parameters(), lr=config.lr_d, betas=(
                config.beta1, config.beta2))

        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=config.lr_g, betas=(
                config.beta1, config.beta2))

        self.train_loader = train_loader
        self.val_loader = val_loader

    def apply_zero_grad(self) -> None:
        """
        Zeros the gradients of the generator and both discriminators.
        """
        self.discriminator_1.zero_grad()
        self.discriminator_2.zero_grad()
        self.generator.zero_grad()

    def enable_gen_disable_disc(self) -> None:
        """
        Enables gradients for the generator while disabling gradients for both discriminators.
        """
        gradients_status(self.generator, True)
        gradients_status(self.discriminator_1, False)
        gradients_status(self.discriminator_2, False)

    def disable_all(self) -> None:
        """
        Disables gradients for the generator and both discriminators.
        """
        gradients_status(self.generator, False)
        gradients_status(self.discriminator_1, False)
        gradients_status(self.discriminator_2, False)

    def train(self) -> None:
        """
        Trains the CycleGAN model, including the generator and both discriminators, with cyclic consistency loss.
        Logs metrics, saves samples, and manages learning rate decay and checkpointing.

        The training loop includes:
        - Discriminator training for real and generated signals.
        - Generator training with identity, GAN, and cyclic consistency losses.
        - Validation at specified intervals.
        - Periodic saving of samples and model checkpoints.
        """
        real_label = 0.9
        progress_bar = tqdm(
            total=self.config.n_iterations //
            self.config.progress_bar_step_iter_size)
        val_set = iter(self.val_loader)
        val_data = next(val_set)
        fixed_mixed_signal = val_data[1].to(self.device)
        fixed_single_signal = val_data[0].to(self.device)
        save_samples(fixed_mixed_signal.detach().cpu().numpy(), 'fixed_mixed')
        save_samples(fixed_single_signal.detach().cpu().numpy(), 'fixed_single')

        gan_model_name = 'gan_cyclic_single_2disc_{}.tar'.format(
            self.config.model_prefix)

        first_iter = 0
        if self.config.take_backup and os.path.isfile(gan_model_name):
            if torch.cuda.is_available():
                checkpoint = torch.load(gan_model_name)
            else:
                checkpoint = torch.load(gan_model_name, map_location='cpu')
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator_1.load_state_dict(checkpoint['discriminator_1'])
            self.discriminator_2.load_state_dict(checkpoint['discriminator_2'])
            self.optimizer_d_1.load_state_dict(checkpoint['optimizer_d_1'])
            self.optimizer_d_2.load_state_dict(checkpoint['optimizer_d_2'])
            self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            self.val_g_cost = checkpoint['val_g_cost']
            self.train_g_cost = checkpoint['train_g_cost']
            self.cyclic_loss = checkpoint['cyclic_loss']
            self.discriminator_loss = checkpoint['discriminator_loss']
            first_iter = checkpoint['n_iterations'] + 1
            for _ in range(0, first_iter, self.config.progress_bar_step_iter_size):
                progress_bar.update()
            if self.writer:
                for index, value in enumerate(self.val_g_cost):
                    self.writer.add_scalar('val_g_cost', value, index)
                for index, value in enumerate(self.train_g_cost):
                    self.writer.add_scalar('train_g_cost', value, index)
                for index, value in enumerate(self.cyclic_loss):
                    self.writer.add_scalar('cyclic_loss', value, index)
                for index, value in enumerate(self.discriminator_loss):
                    self.writer.add_scalar('discriminator_loss', value, index)

            self.generator.eval()
        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()

        target_real = Variable(
            torch.Tensor(
                self.config.batch_size,
                1).fill_(1.0),
            requires_grad=False).to(
            self.device)
        target_fake = Variable(
            torch.Tensor(
                self.config.batch_size,
                1).fill_(0.0),
            requires_grad=False).to(
            self.device)

        generated_single_buffer = ReplayBuffer()
        train_set = iter(self.train_loader)
        for iter_indx in range(first_iter, self.config.n_iterations):
            self.generator.train()
            self.discriminator_1.train()
            self.discriminator_2.train()
            try:
                data = next(train_set)
            except StopIteration:
                train_set = iter(self.train_loader)
                data = next(train_set)

            # in case of unpaired data
            single_signal = data[0].to(self.device)
            mixed_signal = data[1].to(self.device)

            #############################
            # Training First Discriminator
            #############################
            self.apply_zero_grad()
            self.disable_all()
            gradients_status(self.discriminator_1, True)
            generated_single_signal = self.generator(single_signal)
            rest_of_signal = mixed_signal - generated_single_signal
            new_mixed_signal = rest_of_signal + single_signal

            # Real loss
            is_single_signal_r = self.discriminator_1(single_signal)
            # d_loss_real_1 = criterion_GAN(is_single_signal_r,target_real)

            # generated loss
            # generated_single_signal = generated_single_buffer.push_and_pop(generated_single_signal)
            is_single_signal_f = self.discriminator_1(generated_single_signal.detach())
            # d_loss_generated_1 = criterion_GAN(is_single_signal_f, target_fake)

            d_loss_1 = (torch.mean((is_single_signal_r - torch.mean(is_single_signal_f) - target_real) ** 2) + torch.mean(
                (is_single_signal_f - torch.mean(is_single_signal_r) + target_real) ** 2)) / 2  # (d_loss_real_1 + d_loss_generated_1)/2
            d_loss_1.backward()
            self.optimizer_d_1.step()
            #############################
            # Training Second Discriminator
            #############################
            self.apply_zero_grad()
            self.disable_all()
            gradients_status(self.discriminator_2, True)
            # Real loss
            is_mixed_signal_r = self.discriminator_2(mixed_signal)
            # d_loss_real_1 = criterion_GAN(is_mixed_signal_r,target_real)

            # generated loss
            # generated_single_signal = generated_single_buffer.push_and_pop(generated_single_signal)
            is_mixed_signal_f = self.discriminator_2(new_mixed_signal.detach())
            # d_loss_generated_1 = criterion_GAN(is_mixed_signal_f, target_fake)

            d_loss_2 = (torch.mean((is_mixed_signal_r - torch.mean(is_mixed_signal_f) - target_real) ** 2) + torch.mean(
                (is_mixed_signal_f - torch.mean(is_mixed_signal_r) + target_real) ** 2)) / 2  # (d_loss_real_1 + d_loss_generated_1)/2
            d_loss_2.backward()
            self.optimizer_d_2.step()

            #############################
            # Training  generator
            #############################
            self.apply_zero_grad()
            self.enable_gen_disable_disc()

            # Identity loss without it the model would make changes to input even
            # without any need

            identity_loss_1 = criterion_identity(generated_single_signal, single_signal)

            # Gan Loss
            # generated_single_signal = self.generator(mixed_signal)
            is_single_signal_r = self.discriminator_1(single_signal)
            is_single_signal_f = self.discriminator_1(generated_single_signal)
            gan_loss_1 = (
                torch.mean(
                    (is_single_signal_r - torch.mean(is_single_signal_f) + target_real) ** 2) + torch.mean(
                    (is_single_signal_f - torch.mean(is_single_signal_r) - target_real) ** 2)) / 2

            # gan_loss_1 =criterion_GAN(is_single_signal, target_real )

            is_mixed_signal_r = self.discriminator_2(mixed_signal)
            is_mixed_signal_f = self.discriminator_2(new_mixed_signal)
            gan_loss_2 = (
                torch.mean(
                    (is_mixed_signal_r - torch.mean(is_mixed_signal_f) + target_real) ** 2) + torch.mean(
                    (is_mixed_signal_f - torch.mean(is_mixed_signal_r) - target_real) ** 2)) / 2

            # gan_loss_2 = criterion_GAN(is_mixed_signal, target_real )

            reconstructed_single_sinal = self.generator(new_mixed_signal)

            cycle_loss_1 = criterion_cycle(reconstructed_single_sinal, single_signal)
            # Total Loss
            g_cost = identity_loss_1 * 0.5 + \
                (gan_loss_1 + gan_loss_2) + 10 * cycle_loss_1
            g_cost.backward()

            self.cyclic_loss.append(cycle_loss_1)

            self.optimizer_g.step()

            if self.validate and iter_indx % self.config.store_cost_every == 0:
                self.discriminator_loss.append(d_loss_1.item())
                self.train_g_cost.append(g_cost.item())
                if self.writer:
                    self.writer.add_scalar(
                        'discriminator_loss', d_loss_1.item(), iter_indx)
                    self.writer.add_scalar('train_g_cost', g_cost.item(), iter_indx)
                    self.writer.add_scalar('cyclic_loss', cycle_loss_1, iter_indx)
                # validating
                self.disable_all()
                with torch.no_grad():
                    try:
                        val_data = next(val_set)
                    except StopIteration:
                        val_set = iter(self.val_loader)
                        val_data = next(val_set)
                    val_single = val_data[0].to(self.device)
                    val_mixed = val_data[1].to(self.device)
                    val_cost = criterion_GAN(self.discriminator_1(
                        val_single), target_real) + criterion_GAN(self.discriminator_2(val_single), target_real)
                    self.val_g_cost.append(val_cost.item())
                    if self.writer:
                        self.writer.add_scalar('val_g_cost', val_cost.item(), iter_indx)
                    # writer.
                    reconstructed_music = self.generator(val_mixed)
                    self.valid_reconstruction.append(
                        F.mse_loss(
                            reconstructed_music,
                            val_single,
                            reduction='sum').item())

            if iter_indx % self.config.store_cost_every == 0:
                progress_updates = {'Reconstruction': str(
                    self.valid_reconstruction[-1]), 'Loss_D1': str(d_loss_1.item()), 'Loss_g': str(g_cost.item())}
                progress_bar.set_postfix(progress_updates)

            if iter_indx % self.config.progress_bar_step_iter_size == 0:
                progress_bar.update()
            # lr decay
            if self.config.decay_lr:
                decay = max(0.0, 1.0 - (iter_indx * 1.0 / self.config.n_iterations))
                # update the learning rate
                update_optimizer_lr(self.optimizer_d, self.config.lr_d, decay)
                update_optimizer_lr(self.optimizer_g, self.config.lr_g, decay)

            if (iter_indx % self.config.save_samples_every == 0):
                with torch.no_grad():
                    fake = self.generator(fixed_mixed_signal).detach().cpu().numpy()
                save_samples(fake, iter_indx, prefix='predictions')

            if self.config.take_backup and iter_indx % self.config.backup_every_n_iters == 0:
                saving_dict = {
                    'generator': self.generator.state_dict(),
                    'discriminator_1': self.discriminator_1.state_dict(),
                    'optimizer_d_1': self.optimizer_d_1.state_dict(),
                    'discriminator_2': self.discriminator_2.state_dict(),
                    'optimizer_d_2': self.optimizer_d_2.state_dict(),
                    'optimizer_g': self.optimizer_g.state_dict(),
                    'val_g_cost': self.val_g_cost,
                    'train_g_cost': self.train_g_cost,
                    'cyclic_loss': self.cyclic_loss,
                    'discriminator_loss': self.discriminator_loss,
                    'n_iterations': iter_indx
                }
                torch.save(saving_dict, gan_model_name)
