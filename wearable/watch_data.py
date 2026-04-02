import random


def get_watch_data(simulated: bool = True):
    """
    For now this returns simulated physiological data.
    Later you can replace this with real smartwatch values.
    """

    if simulated:
        ecg = random.uniform(-1, 1)
        eda = random.uniform(0, 10)
        temp = random.uniform(28, 38)
        resp = random.uniform(0, 2)
        return ecg, eda, temp, resp

    # future real watch integration goes here
    raise NotImplementedError("Real smartwatch integration not implemented yet.")