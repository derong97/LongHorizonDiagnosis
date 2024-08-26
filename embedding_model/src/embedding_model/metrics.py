import numpy as np
import pandas as pd
from scipy.stats import chi2
from dataclasses import InitVar, dataclass, field

def bootstrappable(func):

    def bootstrap_func(*args, n_bootstrap_samples=None, random_state=None, ci=95, **kwargs):

        estimate = func(*args, **kwargs)

        if n_bootstrap_samples is not None:

            assert type(n_bootstrap_samples) is int, n_bootstrap_samples

            rs = np.random.RandomState(seed=random_state)
            
            N = len(args[0])
            indices = np.arange(N)

            samples = []

            for _ in range(n_bootstrap_samples):

                sample_indices = rs.choice(indices, len(indices), replace=True)

                func_args = [
                    arg[sample_indices]
                    if (hasattr(arg, '__len__') and (len(arg) == N))
                    else arg
                    for arg in args
                ]
                
                func_kwargs = {
                    kw: arg[sample_indices]
                    if (hasattr(arg, '__len__') and (len(arg) == N))
                    else arg
                    for kw, arg in kwargs.items()
                }

                samples.append(func(*func_args, **func_kwargs))

            return (
                estimate,
                np.percentile(samples, 50 - ci / 2., axis=0),
                np.percentile(samples, 50 + ci / 2., axis=0)
            )

        else:

            return estimate

    return bootstrap_func

@bootstrappable
def xAUCt(s_test, t_test, pred_risk, times, pos_group=None, neg_group=None):
    s_test = s_test == 1
    
    # NOTE: enter groups pos_group and neg_group for xAUC_t; omit for AUC_t

    # pred_risk can be 1d (static) or 2d (time-varying)
    if len(pred_risk.shape) == 1:
        pred_risk = pred_risk[:, np.newaxis]

    # positives: s_test = 1 & t_test =< t
    pos = (t_test[:, np.newaxis] <= times[np.newaxis, :]) & s_test[:, np.newaxis]

    if pos_group is not None:
        pos = pos & pos_group[:, np.newaxis]
    
    # negatives: t_test > t
    neg = (t_test[:, np.newaxis] > times[np.newaxis, :])

    if neg_group is not None:
        neg = neg & neg_group[:, np.newaxis]

    valid = pos[:, np.newaxis, :] & neg[np.newaxis, :, :]
    correctly_ranked = valid & (pred_risk[:, np.newaxis, :] > pred_risk[np.newaxis, :, :])

    return np.sum(correctly_ranked, axis=(0, 1)) / np.sum(valid, axis=(0, 1))

@bootstrappable
def xAPt(s_test, t_test, pred_risk, times, pos_group=None, neg_group=None, return_prevalence=False):
    s_test = s_test == 1

    ap = []
    prev = []

    for idx, time in enumerate(times):

        # pred_risk can be 1d (static) or 2d (time-varying)
        if len(pred_risk.shape) == 1:
            prt = pred_risk
        else:
            prt = pred_risk[:, idx]

        recall, precision, threshold, prevalence = xPRt(
            s_test, t_test, prt, time,
            pos_group=pos_group, neg_group=neg_group
        )
        
        ap.append(-1 * np.sum(np.diff(recall) * np.array(precision)[:-1]))
        prev.append(prevalence)

    return (np.array(ap), np.array(prev)) if return_prevalence else np.array(ap)

def xPRt(s_test, t_test, pred_risk, time, pos_group=None, neg_group=None):

    threshold = np.append(np.sort(pred_risk), np.infty)

    # positives: s_test = 1 & t_test =< t
    pos = (t_test < time) & s_test

    if pos_group is not None:
        pos = pos & pos_group
    
    # negatives: t_test > t
    neg = (t_test > time)

    if neg_group is not None:
        neg = neg & neg_group

    # prediction
    pred = pred_risk[:, np.newaxis] > threshold[np.newaxis, :]

    tps = np.sum(pred & pos[:, np.newaxis], axis=0)
    fps = np.sum(pred & neg[:, np.newaxis], axis=0)

    positives = np.sum(pos)
    negatives = np.sum(neg)

    recall = tps / positives
    precision = np.divide(tps, tps + fps, out=np.ones_like(tps, dtype=float), where=(tps + fps) > 0)

    prevalence = positives / (positives + negatives)

    return recall, precision, threshold, prevalence


def hazard_components(s_true, t_true):

    df = (
        pd.DataFrame({'event': s_true, 'time': t_true})
        .groupby('time')
        .agg(['count', 'sum'])
    )

    t = df.index.values
    d = df[('event', 'sum')].values
    c = df[('event', 'count')].values
    n = np.sum(c) - np.cumsum(c) + c

    return t, d, n


def kaplan_meier(s_true, t_true, t_new=None):

    t, d, n = hazard_components(s_true, t_true)

    m = np.cumprod(1 - np.divide(
        d, n,
        out=np.zeros(len(d)),
        where=n > 0
    ))
    
    v = (m ** 2) * np.cumsum(np.divide(
        d, n * (n - d),
        out=np.zeros(len(d)),
        where=n * (n - d) > 0
    ))

    if t_new is not None:
        return interpolate(t, m, t_new)
    else:
        return t, m, v

def interpolate(x, y, new_x, method='pad'):
    
    # s = pd.Series(data=y, index=x)
    # new_y = (s
    #     .reindex(s.index.union(new_x).unique())
    #     .interpolate(method=method)[new_x]
    #     .values
    # )
    
    # return new_y

    return np.interp(new_x, x, y)


def one_calibration(s_test, t_test, tp_onehot, bin_end_times, n_cal_bins=10, return_curves=False):

    N_bins = len(bin_end_times)
    N_pred = len(tp_onehot.T)

    assert N_pred == N_bins or N_pred == N_bins + 1, 'Invalid number of bins'

    cum_test_pred = np.cumsum(tp_onehot, axis=1)[:, :len(bin_end_times)]

    hs_stats = []
    p_vals = []
    times = []
    
    op = []
    ep = []

    for predictions, time in zip(cum_test_pred.T, bin_end_times):

        try:

            prediction_order = np.argsort(-predictions)
            predictions = predictions[prediction_order]
            event_times = t_test.copy()[prediction_order]
            event_indicators = (s_test == 1).copy()[prediction_order]

            # Can't do np.mean since split array may be of different sizes.
            binned_event_times = np.array_split(event_times, n_cal_bins)
            binned_event_indicators = np.array_split(event_indicators, n_cal_bins)
            probability_means = [np.mean(x) for x in np.array_split(predictions, n_cal_bins)]
            
            hosmer_lemeshow = 0
            
            observed_probabilities = []
            expected_probabilities = []
            
            for b in range(n_cal_bins):
                
                prob = probability_means[b]
                
                if prob == 1.0:
                    raise ValueError(
                        "One-Calibration is not well defined: the risk"
                        f"probability of the {b}th bin was {prob}."
                    )
                
                km_model = KaplanMeier(binned_event_times[b], binned_event_indicators[b])
                event_probability = 1 - km_model.predict(time)
                bin_count = len(binned_event_times[b])
                hosmer_lemeshow += (bin_count * event_probability - bin_count * prob) ** 2 / (
                    bin_count * prob * (1 - prob)
                )
                
                observed_probabilities.append(event_probability)
                expected_probabilities.append(prob)

            hs_stats.append(hosmer_lemeshow)
            p_vals.append(1 - chi2.cdf(hosmer_lemeshow, n_cal_bins - 1))
            times.append(time)
            
            op.append(observed_probabilities)
            ep.append(expected_probabilities)

            # return dict(
            #     p_value=1 - chi2.cdf(hosmer_lemeshow, bins - 1),
            #     observed=observed_probabilities,
            #     expected=expected_probabilities,
            # )

        except Exception as e:

            print('Failed for time', time)
            print(e)
            
    if return_curves:
        return np.array(times), np.array(hs_stats), np.array(p_vals), np.array(op), np.array(ep)

    return np.array(times), np.array(hs_stats), np.array(p_vals)


@dataclass
class KaplanMeier:
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities
