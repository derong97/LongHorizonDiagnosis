import torch

class DiscreteFailureTimeNLL(torch.nn.Module):
    
    def __init__(self, bin_boundaries, tolerance=1e-8):
        super(DiscreteFailureTimeNLL,self).__init__()

        self.bin_starts = bin_boundaries[:-1]
        self.bin_ends = bin_boundaries[1:]
        
        self.bin_lengths = self.bin_ends - self.bin_starts
        
        self.tolerance = tolerance
        
    def _discretize_times(self, times):

        return (
            (times[:, None] > self.bin_starts[None, :])
            & (times[:, None] <= self.bin_ends[None, :])
        )

    def _get_proportion_of_bins_completed(self, times):

        return torch.maximum(
            torch.minimum(
                (times[:, None] - self.bin_starts[None, :]) / self.bin_lengths[None, :],
                torch.tensor(1)
            ),
            torch.tensor(0)
        )
    
    def forward(self, predictions, event_indicators, event_times):

        event_likelihood = torch.sum(
            self._discretize_times(event_times) * predictions[:, :-1],
            -1
        ) + self.tolerance

        nonevent_likelihood = 1 - torch.sum(
            self._get_proportion_of_bins_completed(event_times) * predictions[:, :-1],
            -1
        ) + self.tolerance
        
        log_likelihood = event_indicators * torch.log(event_likelihood)
        log_likelihood += (1 - event_indicators) * torch.log(nonevent_likelihood)
        
        return -1. * torch.mean(log_likelihood)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        return cox_ph_loss(log_h, durations, events)


def cox_ph_loss_sorted(log_h: torch.Tensor, events: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum()+eps)


def cox_ph_loss(log_h: torch.Tensor, durations: torch.Tensor, events: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)