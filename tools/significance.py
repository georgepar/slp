import numpy as np
import scipy.stats as stats


def _ztest(mean_scores, null_hyp, std_error, delta=0.0, mode="larger"):
    """Calculate p-value using z-test

    Args:
        mean_scores (float): Sample mean value of our hypothesis (i.e. mean scores of
            group we want to be beter)
        null_hyp (float): Null-hypothesis value. If comparing groups (e.g. A/B test)
            null_hyp is the mean of the first group (the one we do not want to be better)
        std_error (float): Group std error. If comparing groups it's Standard error of
            the difference value1 - value2
        delta (float): The difference between the two groups under the Null hypothesis.
            Leave delta=0.0 unless you know what you are doing.
        mode (str):
            "larger": we test mean_hyp - null_hyp > delta
            "nequal": we test mean_hyp - null_hyp != delta
            "smaller": we test mean_hyp - null_hyp < delta

    Returns:
        (float): Calculated p-value
    """
    p_fn = {
        "smaller": lambda z_score: stats.norm.cdf(z_score),
        "larger": lambda z_score: stats.norm.sf(z_score),
        "nequal": lambda z_score: stats.norm.sf(np.abs(z_score)) * 2,
    }
    z_score = (mean_scores - null_hyp - delta) / std_error
    try:
        p_value = p_fn[mode](z_score)
    except KeyError:
        raise ValueError("mode should be one of [smaller|larger|nequal]")

    return p_value


def _ttest(mean_scores, null_hyp, std_error, dof, delta=0.0, mode="larger"):
    """Calculate p-value using t-test

    Args:
        mean_scores (float): Sample mean value of our hypothesis (i.e. mean scores of
            group we want to be beter)
        null_hyp (float): Null-hypothesis value. If comparing groups (e.g. A/B test)
            null_hyp is the mean of the first group (the one we do not want to be better)
        std_error (float): Group std error. If comparing groups it's Standard error of
            the difference value1 - value2
        dof (int): Degrees of freedom
        delta (float): The difference between the two groups under the Null hypothesis.
            Leave delta=0.0 unless you know what you are doing.
        mode (str):
            "larger": we test mean_hyp - null_hyp > delta
            "nequal": we test mean_hyp - null_hyp != delta
            "smaller": we test mean_hyp - null_hyp < delta

    Returns:
        (float): Calculated p-value
    """
    p_fn = {
        "smaller": lambda t_score: stats.t.cdf(t_score, dof),
        "larger": lambda t_score: stats.t.sf(t_score, dof),
        "nequal": lambda t_score: stats.t.sf(np.abs(t_score), dof) * 2,
    }
    z_score = (mean_scores - null_hyp - delta) / std_error
    try:
        p_value = p_fn[mode](z_score)
    except KeyError:
        raise ValueError("mode should be one of [smaller|larger|nequal]")

    return p_value


def ztest_single(samples, null_hyp, mode="larger"):
    """Calculate p-value when comparing a sample population with a fixed null hypothesis

        samples (list, np.ndarray): List of scores or values
        null_hyp (float): Fixed null hypothesis score
        mode (str):
            "larger": we test mean_hyp > null_hyp
            "nequal": we test mean_hyp != null_hyp
            "smaller": we test mean_hyp < null_hyp

    Returns:
        (float): Calculated p-value
    """
    mu = np.mean(samples)
    stderr = np.std(samples) / np.sqrt(len(samples))

    return _ztest(mu, null_hyp, stderr, delta=0.0, mode=mode)


def ztest_unpaired(samples_b, samples_null, mode="larger"):
    """Calculate p-value when comparing two sample populations

        samples_b (list, np.ndarray): List of scores or values
        samples_null (float): Fixed null hypothesis score
        mode (str):
            "larger": we test mean_hyp > null_hyp
            "nequal": we test mean_hyp != null_hyp
            "smaller": we test mean_hyp < null_hyp

    Returns:
        (float): Calculated p-value
    """
    mu_b = np.mean(samples_b)
    var_b = np.var(samples_b)
    mu_null = np.mean(samples_null)
    var_null = np.var(samples_null)

    stderr = np.sqrt(var_b / (len(samples_b) - 1) + var_null / (len(samples_null) - 1))

    return _ztest(mu_b, mu_null, stderr, delta=0.0, mode=mode)


def ttest_single(samples, null_hyp, mode="larger"):
    """Calculate p-value when comparing a sample population with a fixed null hypothesis

        samples (list, np.ndarray): List of scores or values
        null_hyp (float): Fixed null hypothesis score
        mode (str):
            "larger": we test mean_hyp > null_hyp
            "nequal": we test mean_hyp != null_hyp
            "smaller": we test mean_hyp < null_hyp

    Returns:
        (float): Calculated p-value
    """
    mu = np.mean(samples)
    stderr = np.std(samples) / np.sqrt(len(samples))
    dof = len(samples) - 1

    return _ttest(mu, null_hyp, stderr, dof, delta=0.0, mode=mode)


def ttest_unpaired(samples_b, samples_null, mode="larger"):
    """Calculate p-value when comparing two sample populations

        samples_b (list, np.ndarray): List of scores or values
        samples_null (float): Fixed null hypothesis score
        mode (str):
            "larger": we test mean_hyp > null_hyp
            "nequal": we test mean_hyp != null_hyp
            "smaller": we test mean_hyp < null_hyp

    Returns:
        (float): Calculated p-value
    """
    mu_b = np.mean(samples_b)
    var_b = np.var(samples_b)
    mu_null = np.mean(samples_null)
    var_null = np.var(samples_null)

    sem1 = var_b / (len(samples_b) - 1)
    sem2 = var_null / (len(samples_null) - 1)
    semsum = sem1 + sem2
    z1 = (sem1 / semsum) ** 2 / (len(samples_b) - 1)
    z2 = (sem2 / semsum) ** 2 / (len(samples_null) - 1)
    dof = 1.0 / (z1 + z2)

    stderr = np.sqrt(var_b / (len(samples_b) - 1) + var_null / (len(samples_null) - 1))

    return _ttest(mu_b, mu_null, stderr, dof, delta=0.0, mode=mode)


def significance_test(samples, null_hypothesis, test=None, p_thres=0.05, mode="larger"):
    """Calculate p-value when comparing a sample population with a fixed null hypothesis

        samples (list, np.ndarray): List of scores or values
        null_hypothesis (float, list, np.ndarray): null hypothesis
        test (optional str):
            None: select appropriate test to run
            "t": run t-test
            "z": run z-test
        mode (str):
            "larger": we test mean_hyp > null_hyp
            "nequal": we test mean_hyp != null_hyp
            "smaller": we test mean_hyp < null_hyp

    Returns:
        (float): Calculated p-value
    """
    tests = {
        "t_unpaired": (ttest_unpaired, "Unpaired t-test"),
        "z_unpaired": (ztest_unpaired, "Unpaired z-test"),
        "t_single": (ttest_single, "t-test"),
        "z_single": (ztest_single, "z-test"),
    }

    select_test = ""

    if test is None:
        if len(samples) > 30:
            print("Selecting z-test because n_samples > 30")
            select_test = "z_"
        else:
            print("Selecting t-test because n_samples < 30")
            select_test = "t_"
    else:
        select_test = test + "_"

    if isinstance(null_hypothesis, float) or isinstance(null_hypothesis, int):
        select_test += "single"
    else:
        if len(null_hypothesis) == 1:
            select_test += "single"
        else:
            select_test += "unpaired"

    try:
        test_fn, msg = tests[select_test]
        print(f"Running {msg}")

        p = test_fn(samples, null_hypothesis, mode=mode)
    except KeyError:
        raise ValueError("test should be one of [None|t|z]")

    if p >= p_thres:
        print("Difference is not statistically significant.")
        print(f"p_value={p}>={p_thres}")
    else:
        print("Difference is statistically significant")
        print(f"p_value={p}<{p_thres}")

    return p
