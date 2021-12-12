"""
Course 55812 - Data analysis for decision making - Ex2
Authors:
    Omer Arie Lerinman
    Liat Meir
    Gil Avraham
Date 25/10/2021
"""
# ============================================== Imports =================================================

import numpy as np
from os.path import join, isdir
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from linearmodels.iv.model import IV2SLS
from linearmodels.iv import compare
import pingouin as pg
from sklearn.feature_selection import f_regression

# ============================================== Constants ==============================================


DATA_FILE_NAME = 'WAGE2.DTA'
CUR_DIR = Path(__file__).parent.resolve()


# ============================================== Utils =================================================


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance, 4))
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    # print(f'MSR: {msr:.5f}')
    print(f'MSE: {mse:.5f}')
    print('RMSE: ', round(np.sqrt(mse), 4))


def print_q(question_num: int) -> None:
    """Print the question number"""
    print(f"============= Question {question_num}  =============")


def print_section(section_name: str) -> None:
    """Print the section number"""
    print(f"\n < section {section_name} > ")


def print_standard_error(X: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame, X_names: list):
    """
    calculates and print the standard error.
    """
    N = len(X)
    p = len(X.columns) + 1  # plus one because LinearRegression adds an intercept term
    X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = X.values
    residuals = y_true - y_pred
    ssr = residuals.T @ residuals  # residual_sum_of_squares
    sigma_squared_hat = ssr / (N - p)
    var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
    for i in range(p):
        standard_error = var_beta_hat[i, i] ** 0.5
        if i == 0:
            print(f"Standard error of(beta_hat[0]): {standard_error:.7f}")
        else:
            print(f"Standard error of {X_names[i - 1]} (beta_hat[{i}]): {standard_error:.7f}")


def run_regression(data: pd.DataFrame, y_column_name: str, x_columns_names: list, x_func: list = None,
                   y_func=None, verbose: bool = True, plot: bool = False) -> LinearRegression:
    """
    Calculate and return the results of a regression.
    :param data: input data as a pd.Dataframe
    :param y_column_name: the y attribute
    :param x_columns_names: the x(s) attribute(s)
    :param x_func: optional - perform operation on X columns. If supplied, should be list of functions or Nones, in the
        same length as 'x_columns_names'; default=None.
    :param y_func: optional - perform operation y column. If supplied, should be a (single) function; default=None.
    :param verbose: print models' details; default=True.
    :param plot: boolean indicator if to plot the data; default=False.
    :returns LinearRegression instance
    """
    y = np.array(data[y_column_name])
    x = np.array(data[x_columns_names])
    if y_func is not None:
        y = y_func(y)
    if x_func is not None:
        for column_name, func in zip(x_columns_names, x_func):
            if func is not None:
                x[column_name] = func(x[column_name])
    model = LinearRegression(fit_intercept=True)  # create object for the class
    model.fit(X=x, y=y)  # perform linear regression
    slope = model.intercept_
    Y_pred = model.predict(x)  # make predictions
    if verbose:
        beta_str = '\u03B2'
        x_func_as_list = x_func if x_func is not None else [None for _ in range(len(x_columns_names))]
        functions = [f'{beta_str}{str(x_f)}({i + 1}{x_name})' if x_f is not None else f'{beta_str}{i + 1}{x_name}'
                     for i, (x_f, x_name) in enumerate(zip(x_func_as_list, x_columns_names))]
        right_hand_side = ' + '.join([f'{beta_str}0'] + functions)
        if y_func == np.log:
            left_hand_side = f'log({y_column_name})'
        else:
            left_hand_side = f'{y_func}({y_column_name})' if y_func is not None else f'{y_column_name}'
        print(f'{left_hand_side} = {right_hand_side}')
        functions = [f'{model.coef_[i]:.5f}{str(x_f)}({x_name})' if x_f is not None else f'{model.coef_[i]:.5f}{x_name}'
                     for i, (x_f, x_name) in enumerate(zip(x_func_as_list, x_columns_names))]
        right_hand_side = ' + '.join([f'{slope:.4f}'] + functions)
        print(f'{left_hand_side} = {right_hand_side}')
        if plot and len(x_columns_names) == 1:
            plt.title(f'{y_column_name} as a function of {",".join(x_columns_names)}')
            ax = plt.scatter(x, np.reshape(y, newshape=(-1, 1)), label=y_column_name).axes

            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.plot(x, Y_pred, color='red')
            plt.xlabel(x_columns_names[0] if len(x_columns_names) == 1 else '')
            plt.ylabel(y_column_name)
            plt.grid()
            plt.legend(loc='lower right')
            plt.show()
    return model


def print_cov_table(headers, covAnswers) -> None:
    rowFormat = "{:<12}" * (len(headers) + 1)
    print(rowFormat.format("", *headers))
    for team, row in zip(headers, np.around(covAnswers, 5)):
        print(rowFormat.format(team, *row))

# ============================================== Answers' implementation ===============================

def load_data() -> pd.DataFrame:
    """
    Load and return the data for the exercise.
    :return: A dictionary with file names as keys (without the csv suffix) and pd.DataFrame as values
    """
    data_dir = join(CUR_DIR, 'data')
    assert isdir(data_dir)
    return pd.read_stata(join(data_dir, DATA_FILE_NAME))


def first_question(data: pd.DataFrame) -> None:
    """
    Answer the first question.
    :param data: pd.DataFrame for the exercise
    :return: None
    """
    print(data.describe())

    print_q(1)
    print_section('1')
    model_1 = run_regression(data, 'wage', ['educ'], y_func=np.log)
    beta_0 = model_1.intercept_
    # beta_1 = model_1.coef_[0]
    print_section('2')
    model_2 = run_regression(data, 'wage', ['educ', 'IQ'], y_func=np.log)
    alpha_0 = model_2.intercept_
    alpha_1 = model_2.coef_[0]
    alpha_2 = model_2.coef_[1]
    IQ = data['IQ'].to_numpy()
    education = data['educ'].to_numpy()
    beta_1_est = alpha_1 + (alpha_0 + alpha_2 * IQ - beta_0) / education
    plt.hist(beta_1_est, bins='auto', color='c', edgecolor='k')  # arguments are passed to np.histogram
    plt.axvline(beta_1_est.mean(), color='k', linestyle='dashed', linewidth=1)
    average = beta_1_est.mean()
    plt.title(f"Histogram of beta_1 estimation by the data (av={average})")
    frame1 = plt.gca()
    frame1.axes.yaxis.set_ticklabels([])
    plt.show()
    print_section('3')
    pie12 = np.cov(IQ, education)
    print_cov_table(['IQ', 'Education'], pie12)


def run_OLS(data: pd.DataFrame, y_column_name: str, x_columns_names: list, add_constant: bool = True,
            x_func: list = None, y_func=None) -> RegressionResults:
    """
    Calculate and return the results of a OLS (Ordinary least squares).
    :param data: input data as a pd.Dataframe
    :param y_column_name: the y attribute
    :param x_columns_names: the x(s) attribute(s)
    :param add_constant: indicator to add constant to X; default=True.
    :param x_func: optional - perform operation on X columns. If supplied, should be list of functions or Nones, in the
        same length as 'x_columns_names'; default=None.
    :param y_func: optional - perform operation y column. If supplied, should be a (single) function; default=None.
    :return: an instance of statsmodels.regression.linear_model.RegressionResults
    """
    y = data[y_column_name]
    if y_func is not None:
        y = y_func(y)
    x = data[x_columns_names]
    if x_func is not None:
        for column_name, func in zip(x_columns_names, x_func):
            if func is not None:
                x[column_name] = func(x[column_name])
    if add_constant:
        x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    print(f"\tThe size of X (data sample size): {len(x)}\n\tThe Std: {float(results.bse[0]):.5f}")
    return results


def is_significant(beta_hat: float, beta_hat_std: float, p_value: float = 1.96) -> bool:
    """
    Check if the null hypothesis can be rejected.
    :param beta_hat:  beta hat
    :param beta_hat_std: standard deviation of beta hat
    :param p_value: P value, default=1.96 (5%)
    :return: boolean indicator
    """
    return abs(beta_hat / beta_hat_std) > p_value


def get_confidence_interval(earning_tall: pd.DataFrame, earning_short: pd.DataFrame,
                            normal_curve_area: float = 1.96) -> tuple:
    """
    Calculate and return the confidence interval.
    :param earning_tall: earnings of tall people
    :param earning_short: earnings of short people
    :param normal_curve_area: the approximate value of the 97.5 percentile point of the standard normal distribution -
        95% of the area under a normal curve.
    :return: a tuple of the interval values
    """
    average_short = np.average(earning_short)
    average_tall = np.average(earning_tall)
    n_tall = len(earning_tall)
    n_short = len(earning_short)
    var_tall = np.std(earning_tall) ** 2
    var_short = np.std(earning_short) ** 2
    dif = average_tall - average_short
    std_factor = normal_curve_area * np.sqrt(((((n_tall-1)*var_tall) + (n_short-1)*var_short)/(n_tall + n_short - 2)) *
        ((n_tall + n_short) / (n_tall * n_short)))
    return dif - std_factor, dif + std_factor


def second_question(data: pd.DataFrame) -> None:
    """
    Answer the second question.
    :param fast_food_df: pd.DataFrame of fast food data and related features
    :return: None
    """
    print("\n\n")
    print_q(2)
    print(data.head())

    print_section('a')
    # as a reference -
    run_regression(y_column_name='wage', x_columns_names=['educ', 'exper', 'tenure', 'black'],
                   data=data, y_func=np.log)
    iv_2sls_model = IV2SLS.from_formula("np.log(wage) ~ 1 + exper + tenure + black + [educ ~ sibs]", data).fit()
    print(compare({"2SLS": iv_2sls_model}))
    print(iv_2sls_model)

    # reference check, slightly different results
    data['const_tmp'] = 1
    iv_2sls_model_unadjusted = IV2SLS(np.log(data.wage), data[["const_tmp", "exper", "tenure", "black"]], data.educ,
                                      data.sibs).fit(cov_type="unadjusted")
    print(iv_2sls_model_unadjusted)



    print_section('b')
    educ_reg_model = run_regression(y_column_name='educ', x_columns_names=['sibs', 'exper', 'tenure', 'black'],
                                    data=data, verbose=False)
    educ_hat = educ_reg_model.predict(X=data[['sibs', 'exper', 'tenure', 'black']])
    data['educ_hat'] = educ_hat
    x_variables = ['educ_hat', 'exper', 'tenure', 'black']
    wage_reg_model = run_regression(y_column_name='wage', x_columns_names=x_variables, data=data, y_func=np.log)
    wage_true = np.log(data['wage'])
    X = data[x_variables]
    wage_pred = wage_reg_model.predict(X)
    regression_results(wage_true, wage_pred)
    print_standard_error(X, wage_true, wage_pred, x_variables)

    print_section('c')
    black_data = data[data['black'] == 1]
    black_iv_2sls_model = IV2SLS.from_formula("np.log(wage) ~ 1 + exper + tenure + [educ ~ sibs]", black_data).fit()
    print(compare({"black_iv_2sls_model": black_iv_2sls_model}))
    print(black_iv_2sls_model)
    print(black_iv_2sls_model._f_statistic)

    print_section('d')
    print("\twhen conducting an F-statistic test on sibs:")
    print("\t\tOn all of the data - ")
    ret_val = f_regression(X=data[['sibs', 'exper', 'tenure', 'black']], y=data['educ'])
    print(f'\t\t\tF(1,{len(data)}) ={ret_val[0][0]:.6f}')
    print(f'\t\t\tProb > F = {ret_val[1][0]:.8f}')
    print("\t\tOn the black people data only - ")
    black_data_ret_val = f_regression(X=black_data[['sibs', 'exper', 'tenure']], y=black_data['educ'])
    print(f'\t\t\tF(1,{len(black_data)}) ={black_data_ret_val[0][0]:.6f}')
    print(f'\t\t\tProb > F = {black_data_ret_val[1][0]:.8f}')

    print_section('e')
    non_black_data = data[data['black'] == 0]
    non_black_iv_2sls_model = IV2SLS.from_formula("np.log(wage) ~ 1 + exper + tenure + [educ ~ sibs]", non_black_data).fit()
    print(compare({"non_black_iv_2sls_model": non_black_iv_2sls_model}))
    print(non_black_iv_2sls_model)


if __name__ == '__main__':
    print('55812 Data analysis for decision making\nEx2\nLiat Meir, Omer Arie Lerinman & Gil Avraham \n\n')
    data = load_data()
    first_question(data)
    second_question(data)
    print('Done.')
