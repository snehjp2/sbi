import torch
from torch.distributions import MultivariateNormal

from sbi.inference import NPE_C


def test_npe_c_accepts_custom_loss():
    prior = MultivariateNormal(torch.zeros(1), torch.eye(1))
    inference = NPE_C(prior=prior, show_progress_bars=False)
    theta = prior.sample((10,))
    x = theta + 0.1 * torch.randn_like(theta)

    def custom_loss(
        theta,
        x,
        masks,
        proposal,
        calibration_kernel,
        force_first_round_loss=False,
    ):
        return torch.zeros(theta.size(0), device=theta.device, requires_grad=True)

    posterior_estimator = inference.append_simulations(theta, x).train(
        max_num_epochs=1,
        custom_loss=custom_loss,
    )
    assert posterior_estimator is not None
