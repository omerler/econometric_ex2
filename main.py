"""
Course 55812 - Data analysis for decision making - Ex2
Authors:
    Omer Arie Lerinman, 304919863
    Liat Meir, 305677817
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
import pingouin as pg

# ============================================== Constants ==============================================


DATA_FILE_NAME = 'WAGE2.DTA'
CUR_DIR = Path(__file__).parent.resolve()


# ============================================== Utils =================================================


def print_q(question_num: int) -> None:
    """Print the question number"""
    print(f"============= Question {question_num}  =============")


def print_section(section_name: str) -> None:
    """Print the section number"""
    print(f"section {section_name}")

# ============================================== Answers' implementation ===============================


def scatter_earnings_on_height(earnings_df: pd.DataFrame) -> None:
    """
    Creates and plots a scatter-plot of annual earnings (Earnings) on height (Height)
    :param earnings_df: pd.DataFrame of earnings (and more)
    :return: None
    """
    ax = plt.scatter(x=earnings_df['height'], y=earnings_df['earnings'], c="g", alpha=0.5, marker='o',
                     label="Earnings").axes
    plt.ylabel("Earnings (average earnings bucket)")
    plt.xlabel("Height (Inches)")
    plt.title('Earnings on height regression')
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(join(CUR_DIR, 'earnings_on_height.png'))
    plt.show()


def load_data() -> dict:
    """
    Load and return the data for the exercise.
    :return: A dictionary with file names as keys (without the csv suffix) and pd.DataFrame as values
    """
    data_dir = join(CUR_DIR, 'data')
    assert isdir(data_dir)
    return pd.read_stata(join(data_dir, DATA_FILE_NAME))


def run_regression(data: pd.DataFrame, y_column_name: str, x_columns_names: list, x_func: list = None,
                   y_func=None) -> LinearRegression:
    """
    Calculate and return the results of a regression.
    :param data: input data as a pd.Dataframe
    :param y_column_name: the y attribute
    :param x_columns_names: the x(s) attribute(s)
    :param x_func: optional - perform operation on X columns. If supplied, should be list of functions or Nones, in the
        same length as 'x_columns_names'; default=None.
    :param y_func: optional - perform operation y column. If supplied, should be a (single) function; default=None.
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
    print(f'\tThe estimated slope: {slope:.4f}')
    beta_str = '\u03B2'
    x_func_as_list = x_func if x_func is not None else [None for _ in range(len(x_columns_names))]
    functions = [f'{beta_str}{str(x_f)}({i+1}{x_name})' if x_f is not None else f'{beta_str}{i+1}{x_name}'
                 for i,(x_f, x_name) in enumerate(zip(x_func_as_list, x_columns_names))]
    right_hand_side = ' + '.join([f'{beta_str}0'] + functions)
    if y_func == np.log:
        left_hand_side = f'log({y_column_name})'
    else:
        left_hand_side = f'{y_func}({y_column_name})' if y_func is not None else f'{y_column_name}'
    print(f'{left_hand_side} = {right_hand_side}')
    functions = [f'{model.coef_[i]:.5f}{str(x_f)}({x_name})' if x_f is not None else f'{model.coef_[i]:.5f}{x_name}'
                 for i,(x_f, x_name) in enumerate(zip(x_func_as_list, x_columns_names))]
    right_hand_side = ' + '.join([f'{slope:.4f}'] + functions)
    print(f'{left_hand_side} = {right_hand_side}')
    if len(x_columns_names)==1:
        plt.title(f'{y_column_name} as a function of {",".join(x_columns_names)}')
        ax = plt.scatter(x, np.reshape(y, newshape=(-1,1)), label=y_column_name).axes

        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.plot(x, Y_pred, color='red')
        plt.xlabel(x_columns_names[0] if len(x_columns_names)==1 else '')
        plt.ylabel(y_column_name)
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()
    return model


def first_question(data: pd.DataFrame) -> None:
    """
    Answer the first question.
    :param data: pd.DataFrame for the exercise
    :return: None
    """
    print(data.describe())

    print_q(1)
    print_section('1.')
    model_1 = run_regression(data, 'wage', ['educ'], y_func=np.log)
    beta_0 = model_1.intercept_
    beta_1 = model_1.coef_[0]
    print_section('2.')
    model_2 = run_regression(data, 'wage', ['educ', 'IQ'], y_func=np.log)
    alpha_0 = model_2.intercept_
    alpha_1 = model_2.coef_[0]
    alpha_2 = model_2.coef_[1]
    IQ = data['IQ'].to_numpy()
    education = data['educ'].to_numpy()
    beta_1_est = alpha_1 + (alpha_0 + alpha_2*IQ - beta_0) / education
    plt.hist(beta_1_est, bins='auto', color='c', edgecolor='k') # arguments are passed to np.histogram
    plt.axvline(beta_1_est.mean(), color='k', linestyle='dashed', linewidth=1)
    average = beta_1_est.mean()
    plt.title(f"Histogram of beta_1 estimation by the data (av={average})")
    frame1 = plt.gca()
    frame1.axes.yaxis.set_ticklabels([])
    plt.show()


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
    return abs(beta_hat/beta_hat_std) > p_value


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
                                             ((n_tall + n_short) / (n_tall*n_short)))
    return dif - std_factor, dif + std_factor


def second_question(data: pd.DataFrame) -> None:
    """
    Answer the second question.
    :param fast_food_df: pd.DataFrame of fast food data and related features
    :return: None
    """

    print("\n\n")
    print_q(2)

    print_section('a')
    # todo


if __name__ == '__main__':
    print('55812 Data analysis for decision making\nEx2\nLiat Meir (305677817) & Omer Arie Lerinman (304919863)\n\n')
    data = load_data()
    first_question(data)
    second_question(data)
    print('Done.')

