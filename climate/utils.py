import pandas as pd
import numpy as np
import os


def load_all_data(dir, write=False):
    files = os.listdir(dir)

    df = pd.DataFrame()
    for file in files:
        if "projections_FAIR_" in file:
            temp = pd.read_csv(dir + '/' + file)
            temp['ssp'] = file[-10:-4]
            df = pd.concat([df, temp])
    df['ice_source'] = df['ice_source'].apply(lambda x: x.lower())
    if write:
        df.to_csv(dir + '/' + 'all_ssp.csv')
    return df


def get_FAIR(dir=f'../data/Edwards/proj_MAIN_2100', source='glaciers', dataset='all', region=None):
    check_input(source, ['gris', 'glaciers', 'ais'])
    check_input(dataset, ['all', 'ssp370', 'ssp245', 'ssp126', 'ssp119', 'ssp585', 'sspndc'])

    data = load_all_data(dir=dir)
    data = data[data['ice_source'] == source]
    if dataset != 'all':
        data = data[data.ssp == dataset]
    if source == "glaciers":
        if region is not None:
            data = data[data.region == 'region_' + str(region)]

        x = np.array(data.GSAT)
        y = np.array(data.SLE)

    if source == "gris":
        x = np.zeros((len(data), 2))
        x[:, 0] = np.array(data.GSAT)
        x[:, 1] = np.array(data['melt'])
        y = np.array(data.SLE)

    if source == "ais":
        x = np.zeros((len(data), 3))
        x[:, 0] = np.array(data.GSAT)
        x[:, 1] = np.array(data['melt'])
        x[:, 2] = np.array(data['collapse'])
        y = np.array(data.SLE)

    return x, y, data


def linear_fit(X, y=None):
    if y is None:
        assert X.shape[1] == 2, f'If y is not provided, X.shape[1] == 2. Currently: {X.shape[1]}'
        y = X[:, 1]
        X = X[:, 0]

    try:
        X.shape[1] == 1
    except IndexError:
        X = X.squeeze()

    fit = np.polyfit(X, y, 1)
    x_line = X
    y_line = np.polyval(fit, x_line)

    return x_line, y_line, fit


def check_input(input, options):
    input = input.lower()
    assert input in options, f"input must be in {options}, received {input}"
