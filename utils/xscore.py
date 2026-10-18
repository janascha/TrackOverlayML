import numpy as np


def getxScore(name, scores, threshold, eventSet, **kwargs):

    # Extract kwargs: there's probably a cleaner way of doing things.
    if "frac" in kwargs:
        frac = kwargs["frac"]
    else:
        frac = 0.25

    if "estimator" in kwargs:
        estimator = kwargs["estimator"]
    else:
        estimator = 50

    if "event_mixing" in kwargs:
        event_mixing = kwargs["event_mixing"]
    else:
        event_mixing = "sum"

    if name.casefold() == "none":
        return 0

    elif name.casefold() == "simpleratio":
        badTracks = np.count_nonzero(scores > threshold, axis=1)
        multiplicity = np.ma.count(scores, axis=1)
        rouletteScore = badTracks / multiplicity
        return np.squeeze(rouletteScore.data)

    elif name.casefold() == "average":
        # I want to take mean of truth tracks which are bad, i.e. classifier predicts that there
        # is a difference in the with and without pu case.
        # There's a smaller than sign because we want to mask the truth tracks that are 'good'
        rouletteScore = np.ma.masked_where(scores < threshold, scores).mean(axis=1)
        return np.squeeze(rouletteScore.data)

    elif name.casefold() == "sum":
        # I want to take sum of truth tracks which are bad, i.e. classifier predicts that there
        # is a difference in the with and without pu case.
        rouletteScore = np.ma.masked_where(scores < threshold, scores).sum(axis=1)
        return np.squeeze(rouletteScore.data)

    elif name.casefold() == "prod":
        # I want to take prod of truth tracks which are bad, i.e. classifier predicts that there
        # is a difference in the with and without pu case.
        rouletteScore = np.ma.masked_where(scores < threshold, scores).prod(axis=1)
        return np.squeeze(rouletteScore.data)

    elif name.casefold() == "max":
        # I want to take max of truth tracks which are bad, i.e. classifier predicts that there
        # is a difference in the with and without pu case.
        rouletteScore = np.ma.masked_where(scores < threshold, scores).max(axis=1)
        return np.squeeze(rouletteScore.data)

    elif name.casefold() == "min":
        # I want to take min of truth tracks which are bad, i.e. classifier predicts that there
        # is a difference in the with and without pu case.
        rouletteScore = np.ma.masked_where(scores < threshold, scores).min(axis=1)
        return np.squeeze(rouletteScore.data)

    elif name.casefold() == "sumpt":
        event_sum_Pt = np.array(eventSet[:, :, 4], dtype=float)
        event_sum_Pt = np.ma.masked_invalid(event_sum_Pt).sum(axis=1)
        return event_sum_Pt.data

    #elif name.casefold() == "jetpt":
        #event_lj_Pt = np.array(eventSet[:, 0, 7], dtype=float)
        #event_lj_Pt = np.ma.masked_invalid(event_lj_Pt)
        #return event_lj_Pt.data

    elif name.casefold() == "density_2":
        density_2 = np.array(eventSet[:, :, 5], dtype=float)
        if event_mixing.casefold() == "max":
            max_density_2 = np.ma.masked_invalid(density_2).max(axis=1)
            return max_density_2
        elif event_mixing.casefold() == "min":
            min_density_2 = np.ma.masked_invalid(density_2).min(axis=1)
            return min_density_2
        elif event_mixing.casefold() == "prod":
            prod_density_2 = np.ma.masked_invalid(density_2).prod(axis=1)
            return prod_density_2
        elif event_mixing.casefold() == "sum":
            sum_density_2 = np.ma.masked_invalid(density_2).sum(axis=1)
            return sum_density_2
        elif event_mixing.casefold() == "avg":
            avg_density_2 = np.ma.masked_invalid(density_2).mean(axis=1)
            return avg_density_2
        elif event_mixing.casefold() == "ptweight":
            wt_density_2 = np.ma.masked_invalid(density_2)*np.ma.masked_invalid(eventSet[:,:,4])
            wt_density_2 = wt_density_2.sum(axis=1)
            return wt_density_2

    elif name.casefold() == "density_5":
        density_5 = np.array(eventSet[:, :, 6], dtype=float)
        if event_mixing.casefold() == "max":
            max_density_5 = np.ma.masked_invalid(density_5).max(axis=1)
            return max_density_5
        elif event_mixing.casefold() == "min":
            min_density_5 = np.ma.masked_invalid(density_5).min(axis=1)
            return min_density_5
        elif event_mixing.casefold() == "prod":
            prod_density_5 = np.ma.masked_invalid(density_5).prod(axis=1)
            return prod_density_5
        elif event_mixing.casefold() == "sum":
            sum_density_5 = np.ma.masked_invalid(density_5).sum(axis=1)
            return sum_density_5
        elif event_mixing.casefold() == "avg":
            avg_density_5 = np.ma.masked_invalid(density_5).mean(axis=1)
            return avg_density_5
        elif event_mixing.casefold() == "ptweight":
            wt_density_5 = np.ma.masked_invalid(density_5)*np.ma.masked_invalid(eventSet[:,:,7])
            wt_density_5 = wt_density_5.sum(axis=1)
            return wt_density_5

    elif name.casefold() == "random":
        nevents = eventSet.shape[0]
        to_fast = int(frac * nevents)
        to_full = nevents - to_fast
        score = np.concatenate([np.repeat(2, to_fast), np.repeat(4, to_full)])
        score_collect = [np.random.permutation(score) for i in range(estimator)]

        return np.array(score_collect)

    else:
        raise NotImplementedError
