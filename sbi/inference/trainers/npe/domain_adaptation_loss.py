from typing import Callable

import torch
from geomloss import SamplesLoss
from torch import Tensor, nn

from sbi.inference.trainers.npe.npe_base import PosteriorEstimatorTrainer
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event


def make_sinkhorn_loss(trainer: "PosteriorEstimatorTrainer") -> Callable:
    """Return a custom loss for domain adaptation using Sinkhorn divergence.

    The returned function expects that each training batch contains simulation
    outputs ``x`` where the first half corresponds to the source domain
    (paired with ``theta``) and the second half to the target domain. Only the
    source samples are used for the log-probability term.

    When calling :func:`append_simulations`, ``theta`` must have the same length
    as the concatenated ``x``. A single batch of parameters therefore has to be
    duplicated prior to the call, for example:

    ```python
    x = torch.cat([x_source, x_target])
    theta = torch.cat([theta, theta])
    inference.append_simulations(theta, x)
    ```

    Two trainable scaling factors ``eta_1`` and ``eta_2`` are instantiated and
    automatically appended to the trainer's optimizer so that they are updated
    during training. Both parameters are placed on the same device as the
    trainer.
    """

    device = trainer._device
    eta_1 = nn.Parameter(torch.tensor(1.0, device=device))
    eta_2 = nn.Parameter(torch.tensor(1.0, device=device))

    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        trainer.optimizer.add_param_group({"params": [eta_1, eta_2]})
    sinkhorn_fn: Callable[[Tensor, Tensor, float], Tensor]

    def custom_loss(
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal,
        calibration_kernel: Callable[[Tensor], Tensor],
        force_first_round_loss: bool = False,
    ) -> Tensor:
        nonlocal sinkhorn_fn

        batch_size = theta.shape[0]
        source_x = x[:batch_size]
        target_x = x[batch_size:]
        masks = masks[:batch_size]

        theta_b = reshape_to_batch_event(
            theta, event_shape=trainer._neural_net.input_shape
        )
        source_x_b = reshape_to_batch_event(
            source_x, event_shape=trainer._neural_net.condition_shape
        )

        if trainer._round == 0 or force_first_round_loss:
            base_loss = trainer._neural_net.loss(theta_b, source_x_b)
        else:
            base_loss = -trainer._log_prob_proposal_posterior(
                theta, source_x, masks, proposal
            )

        src_feat = trainer._neural_net.embed_condition(source_x_b)
        tgt_feat = trainer._neural_net.embed_condition(
            reshape_to_batch_event(target_x, trainer._neural_net.condition_shape)
        )

        with torch.no_grad():
            max_distance = torch.max(torch.cdist(src_feat, tgt_feat, p=2))
            blur = max(0.05 * max_distance.item(), 0.01)
            sinkhorn_fn = SamplesLoss("sinkhorn", blur=blur, scaling=0.9)

        da_loss = sinkhorn_fn(src_feat, tgt_feat)

        total = (
            (1.0 / (2 * eta_1**2)) * base_loss
            + (1.0 / (2 * eta_2**2)) * da_loss
            + torch.log(torch.abs(eta_1) * torch.abs(eta_2))
        )

        return calibration_kernel(source_x) * total

    return custom_loss
