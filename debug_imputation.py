import os

from IPython.display import set_matplotlib_formats
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from jax import numpy as jnp
from jax import ops, random
from jax.scipy.special import expit

import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, Predictive

plt.style.use("seaborn")
# if "NUMPYRO_SPHINXBUILD" in os.environ:
#    set_matplotlib_formats("svg")

from icecream import ic


def get_normalization_infos(*x_s, columns):
    normalization_infos = pd.DataFrame(data=[[1000 for _ in range(len(columns))], [0 for _ in range(len(columns))]],
                                       index=["min", "max"],
                                       columns=columns)
    for x in x_s:
        for col in columns:
            min_value = min(normalization_infos[col]["min"], x[col].min())
            max_value = max(normalization_infos[col]["max"], x[col].max())
            normalization_infos[col] = [min_value, max_value]

    normalization_infos.loc["spread"] = normalization_infos.apply(lambda c: c["max"] - c["min"], axis=0)

    return normalization_infos


def normalize(x: pd.DataFrame, normalization_infos: pd.DataFrame):
    for col in x.columns:
        x[col] = (x[col] - normalization_infos[col]["min"]) / normalization_infos[col]["spread"]
    return x


def create_nans(dataset, columns, ratio_nan):
    for col in columns:
        random_vec = np.random.random(dataset[col].shape) < 1 - ratio_nan
        dataset[col] = dataset[col].where(random_vec, other=np.nan)
    return dataset


dataset = pd.read_csv("../data/X_station_day.csv")
del dataset['station_id']
ground_truth = dataset.ground_truth.values
del dataset['ground_truth']

columns = dataset.columns.tolist()
print(columns)

normalisation_infos = get_normalization_infos(dataset, columns=columns)
dataset = normalize(dataset, normalisation_infos)

# column_to_impute = ['wind_direction', 'temperature', 'humidity', 'dew_point', 'precipitations']
column_to_impute = ['wind_direction']
dataset = create_nans(dataset, column_to_impute, 0.001)

print(dataset['precipitations'].isna().sum())

"""print(dataset.dew_point.values[0])
dataset.dew_point.values[0] = np.nan
print(dataset.dew_point.values[0])"""

"""dew_point_mu = dataset['dew_point'].mean()
dew_point_sigma = dataset['dew_point'].std()

print(dew_point_mu, dew_point_sigma)"""

data = dict(
    latitude=dataset.latitude.values,
    longitude=dataset.longitude.values,
    altitude=dataset.altitude.values,
    wind_direction=dataset.wind_speed.values,
    temperature=dataset.temperature.values,
    humidity=dataset.humidity.values,
    dew_point=dataset.dew_point.values,
    precipitations=dataset.precipitations.values,
)

mu_col = dict()
sigma_col = dict()

for column in column_to_impute:
    mu_col[column] = dataset[column].mean()
    sigma_col[column] = dataset[column].std()

print(mu_col)
print(sigma_col)


# year = dataset.year.values,
# month = dataset.month.values,
# day = dataset.day.values,

def model_dew_point_imputation_with_temperature(
        latitude, longitude, altitude, wind_direction, temperature, humidity, dew_point, precipitations, mu, sigma,
        ground_truth=None,
        nan_columns=None
):
    lat, long, alt, w_d, temp, hum, d_pt, prec = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    default = numpyro.sample("default", dist.Normal(0.0, 0.2))
    if latitude is not None:
        bayes_latitude = numpyro.sample('bayes_latitude', dist.Normal(0, 1))
        lat = bayes_latitude * latitude
    if longitude is not None:
        bayes_longitude = numpyro.sample('bayes_longitude', dist.Normal(0, 1))
        long = bayes_longitude * longitude
    if altitude is not None:
        bayes_altitude = numpyro.sample('bayes_altitude', dist.Normal(0, 1))
        alt = bayes_altitude * altitude

    if 'wind_direction' in nan_columns:
        wd_mu = numpyro.sample("wd_mu", dist.Normal(mu['wind_direction'], 0.2))
        wd_sigma = numpyro.sample("wd_sigma", dist.Normal(sigma['wind_direction'], 0.2))

        wind_direction_nanidx = np.array(np.isnan(wind_direction).astype(int)).nonzero()[0]
        wd_impute = numpyro.sample("wd_impute", dist.Normal(wd_mu, wd_sigma)
                                   .expand((len(wind_direction_nanidx),))
                                   .mask(False))

        wind_direction = ops.index_update(wind_direction, wind_direction_nanidx, wd_impute)

        numpyro.sample("latent_wind_direction", dist.Normal(mu['wind_direction'], sigma['wind_direction']),
                       obs=wind_direction)
        bayes_wind_direction = numpyro.sample("bayes_wind_direction", dist.Normal(0, 1))
        w_d = bayes_wind_direction * wind_direction
    else:
        if wind_direction is not None:
            bayes_wind_direction = numpyro.sample('bayes_wind_direction', dist.Normal(0, 1))
            w_d = bayes_wind_direction * wind_direction

    if 'temperature' in nan_columns:
        temp_mu = numpyro.sample("temp_mu", dist.Normal(mu['temperature'], 0.2))
        temp_sigma = numpyro.sample("temp_sigma", dist.Normal(sigma['temperature'], 0.2))

        temperature_nanidx = np.array(np.isnan(temperature).astype(int)).nonzero()[0]
        temperature_impute = numpyro.sample("temperature_impute", dist.Normal(temp_mu, temp_sigma)
                                            .expand((len(temperature_nanidx),))
                                            .mask(False))

        temperature = ops.index_update(temperature, temperature_nanidx, temperature_impute)

        numpyro.sample("latent_temperature", dist.Normal(mu['temperature'], sigma['temperature']), obs=temperature)
        bayes_temperature = numpyro.sample("bayes_temperature", dist.Normal(0, 1))
        temp = bayes_temperature * temperature
    else:
        if temperature is not None:
            bayes_temperature = numpyro.sample('bayes_temperature', dist.Normal(0, 1))
            temp = bayes_temperature * temperature

    if 'humidity' in nan_columns:
        hum_mu = numpyro.sample("hum_mu", dist.Normal(mu['humidity'], 0.2))
        hum_sigma = numpyro.sample("hum_sigma", dist.Normal(sigma['humidity'], 0.2))

        humidity_nanidx = np.array(np.isnan(humidity).astype(int)).nonzero()[0]
        humidity_impute = numpyro.sample("humidity_impute", dist.Normal(hum_mu, hum_sigma)
                                         .expand((len(humidity_nanidx),))
                                         .mask(False))

        humidity = ops.index_update(humidity, humidity_nanidx, humidity_impute)

        numpyro.sample("latent_humidity", dist.Normal(mu['humidity'], sigma['humidity']), obs=humidity)
        bayes_humidity = numpyro.sample('bayes_humidity', dist.Normal(0, 1))
        hum = bayes_humidity * humidity
    else:
        if humidity is not None:
            bayes_humidity = numpyro.sample('bayes_humidity', dist.Normal(0, 1))
            hum = bayes_humidity * humidity

    if 'dew_point' in nan_columns:
        dew_point_mu = numpyro.sample("dew_point_mu", dist.Normal(mu['dew_point'], 0.2))
        dew_point_sigma = numpyro.sample("dew_point_sigma", dist.Normal(sigma['dew_point'], 0.2))

        dew_point_nanidx = np.array(np.isnan(dew_point).astype(int)).nonzero()[0]
        dew_point_impute = numpyro.sample("dew_point_impute", dist.Normal(dew_point_mu, dew_point_sigma)
                                          .expand((len(dew_point_nanidx),))
                                          .mask(False))

        dew_point = ops.index_update(dew_point, dew_point_nanidx, dew_point_impute)

        numpyro.sample("latent_dew_point", dist.Normal(mu['dew_point'], sigma['dew_point']), obs=dew_point)
        bayes_dew_point = numpyro.sample('bayes_dew_point', dist.Normal(0, 1))
        d_pt = bayes_dew_point * dew_point
    else:
        if dew_point is not None:
            bayes_dew_point = numpyro.sample('bayes_dew_point', dist.Normal(0, 1))
            d_pt = bayes_dew_point * dew_point
    if 'precipitations' in nan_columns:
        precipitations_mu = numpyro.sample("precipitations_mu", dist.Normal(mu['precipitations'], 0.2))
        precipitations_sigma = numpyro.sample("precipitations_sigma", dist.Normal(sigma['precipitations'], 0.2))

        precipitations_nanidx = np.array(np.isnan(precipitations).astype(int)).nonzero()[0]
        precipitations_impute = numpyro.sample("precipitations_impute",
                                               dist.Normal(precipitations_mu, precipitations_sigma)
                                               .expand((len(precipitations_nanidx),))
                                               .mask(False))

        precipitations = ops.index_update(precipitations, precipitations_nanidx, precipitations_impute)

        numpyro.sample("latent_precipitations", dist.Normal(mu['precipitations'], sigma['precipitations']),
                       obs=precipitations)
        bayes_precipitations = numpyro.sample('bayes_precipitations', dist.Normal(0, 1))
        prec = bayes_precipitations * precipitations
    else:
        if precipitations is not None:
            bayes_precipitations = numpyro.sample('bayes_precipitations', dist.Normal(0, 1))
            prec = bayes_precipitations * precipitations

    sigma_model = numpyro.sample("sigma", dist.Exponential(1.0))
    mu_model = default + lat + long + alt + w_d + temp + hum + d_pt + prec
    # print("sigma", sigma_model, "mu", mu_model)
    numpyro.sample("ground_truth", dist.Normal(mu_model, sigma_model), obs=ground_truth)


# WITH NaNS (ratio 0.4)
mcmc = MCMC(NUTS(model_dew_point_imputation_with_temperature), num_warmup=1000, num_samples=1000)
mcmc.run(random.PRNGKey(0), **data, ground_truth=ground_truth, nan_columns=column_to_impute, mu=mu_col, sigma=sigma_col)
# Print the statistics of posterior samples collected during running this MCMC instance. (documentation)
mcmc.print_summary()
