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
    left_hand_side = f'log({y_column_name})' if y_func is not None else f'{y_column_name}'
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


def run_gender_regression(earnings_df: pd.DataFrame, gender_enum: int) -> None:
    """
    Run a regression on the Earnings on Height on a specific gender and print results.
    :param earnings_df: pd.DataFrame of earnings and related features
    :param gender_enum: 0 for female, 1 for male
    :return: None
    """
    gender_str = GENDER_TO_TEXT[gender_enum]
    print(f'Run regression of Earnings on Height on {gender_str}...')
    earnings = earnings_df[earnings_df['sex'] == gender_enum]['earnings'].to_numpy()
    height = earnings_df[earnings_df['sex'] == gender_enum]['height'].to_numpy().reshape(-1,1)
    average_height = np.average(height)
    model = LinearRegression()
    model.fit(X=height, y=earnings)
    slope = float(model.coef_[0])
    print(f'\tEstimated slope: {slope:.4f} .')
    average_female_earnings = model.predict([[average_height]])
    assert average_female_earnings == np.average(earnings)
    print(f'\tThe slope value means that the predicted response rises by {slope:.4f} when 𝑥 is increased by one.'
          f'\n\tThus, A randomly selected {gender_str} who is 1 inch taller than the average {gender_str} in the sample '
          f'(with height of {float(average_height):,.4f} and earnings of {float(average_female_earnings):,.4f}) would get {slope:.4f} more dollars,'
          f'\n\twhich is {float(average_female_earnings + slope):,.4f}$ in total.')


def first_question(data: pd.DataFrame) -> None:
    """
    Answer the first question.
    :param data: pd.DataFrame for the exercize
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
    exit()
    median_height = np.median(heights)
    print(f'\tThe median value of the height is {median_height:.4f} Inches.')
    earning_short = earnings_df[heights <= 67]['earnings']
    earning_tall = earnings_df[heights > 67]['earnings']
    average_short = np.average(earning_short)
    print(f'\tThe average earnings for workers whose height is at most 67 Inches: {average_short:,.4f}$.')
    average_tall = np.average(earning_tall)
    print(f'\tThe average earnings for workers whose height is greater than 67 Inches: {average_tall:,.4f}$.')
    if average_tall > average_short:
        print(f'\tOn average, taller workers earn more than shorter workers by {average_tall - average_short:,.4f}$.')
    elif average_tall < average_short:
        print(f'\tOn average, shorter workers earn more than taller workers by {average_tall - average_short:,.4f}$.')
    else:
        print(f'\tOn average, taller workers earn the same as shorter workers ({average_short:,.4f}$ for both).')
    model_results = run_OLS(earnings_df, 'earnings', ['height'], add_constant=True)
    print(f"\tThe Model Results:\n{model_results.summary()}\n")
    _, beta_height = model_results.params
    _, std_height = model_results.bse
    interval_low, interval_high = get_confidence_interval(earning_tall, earning_short)
    print(f"\tThe confidence interval for the height difference is [{interval_low:,.4f} to {interval_high:,.4f}].")
    import statsmodels.stats.api as sms
    control_low, control_high = sms.CompareMeans(sms.DescrStatsW(earning_tall),
                                                 sms.DescrStatsW(earning_short)).tconfint_diff(usevar='unequal')
    print(f"\tcontrol test: confidence interval (based on statsmodels.stats.api): "
          f"[{control_low:,.4f} to {control_high:,.4f}]")
    # T-test
    pingouin_results = pg.ttest(earning_tall, earning_short)
    print(f"\tcontrol test: confidence interval (based on pingouin.ttest): {pingouin_results['CI95%']}")

    print_section('b')
    scatter_earnings_on_height(earnings_df)
    print('\t The points are clustered into distinct horizontal lines because the earnings were grouped into buckets\n'
          '\t e.g. 23 ranges of earnings.')

    print_section('c')  # The linear regression
    run_regression(earnings_df)

    print_section('d')
    model = LinearRegression()
    y = np.array(earnings_df['earnings'])
    x = np.array(heights * INCHES_TO_CM_FACTOR).reshape(-1,1)
    print("Run a regression of the earning on height in centimeter...")
    model.fit(X=x, y=y)
    print(f'\tEstimated slope: {model.coef_[0]:.4f}')
    print(f'\tEstimated intercept: {model.intercept_:.4f}, which is equal to the intercept we got in the regression on'
          f' inches, as expected.')
    print(f'\tThe R^2 score is {model.score(X=x, y=y):.4f}')
    # calculate the standard error:
    x = sm.add_constant(x)
    model_d = sm.OLS(y, x)
    model_results_d = model_d.fit()
    _, std_height = model_results_d.bse
    print(f'\tThe standard error of the slope is {std_height:.4f}')

    print_section('e')
    run_gender_regression(earnings_df, FEMALE)

    print_section('f')
    run_gender_regression(earnings_df, MALE)

    print_section('g')
    print('\tWe saw that the height is correlated with earning. Thus, it is plausible that other factors that '
          '\n\tcause earning increase will be correlated with the height in some manner.')


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
    black_proportion = fast_food_df['prpblck']
    income = fast_food_df['income']
    average_income = np.nanmean(income)
    average_black = np.nanmean(black_proportion)
    std_black = np.nanstd(black_proportion)
    std_income = np.nanstd(income)

    print(f'\tThe average income is {average_income:,.4f}$ with std={std_income:,.4f}')
    print(f'\tThe average proportion of black people in the zip code area is {average_black:,.4f}, '
          f'in units of normalized proportion of the \n\twhole population, with std={std_black:,.4f}')

    print_section('b')
    fast_food_df = fast_food_df[['prpblck', 'income', 'psoda', 'prppov']]
    fast_food_df = fast_food_df.dropna()
    beta_str = '\u03B2'

    model_results = run_OLS(fast_food_df, 'psoda', ['prpblck', 'income'], add_constant=True)
    beta_const, beta_prpblck, beta_income = model_results.params

    print(f"\tpsode = {beta_str}0 + {beta_str}1 * prpblck + {beta_str}2 * income + u")
    print(f"\tpsode = {beta_const:,.4f} + {beta_prpblck:,.4f}*prpblck + {beta_income:,.4f}*income + u")
    print(f"\tThe sample size of the regression is {len(fast_food_df['prpblck'])}.")
    print(f"\tThe R^2 of the regression is {model_results.rsquared:.4f}.")
    print(f"\tThe 'prpblck' coefficient of the regression represent the linear correlation between the density of black"
          f"\n\t people in the specific area with the price of Soda. As it represents {beta_prpblck / beta_const:.4f} of "
          f"\n\t {beta_str}0, I believe this coefficient is an economically significant effect.")

    print_section('c')
    model_results2 = run_OLS(fast_food_df, 'psoda', ['prpblck'], add_constant=True)
    beta_const2, beta_prpblck2 = model_results2.params
    print(f"\tWhen removing the income, the OLS results are:")
    print(f"\tpsode = {beta_const2:,.4f} + {beta_prpblck2:,.4f}*prpblck + u")
    print(f"\tThe effect of 'prpblck' is higher when we control for income.")

    print_section('d')
    model_results3 = run_OLS(fast_food_df, 'psoda', ['prpblck', 'income'], add_constant=True, y_func=np.log,
                             x_func=[None, np.log])
    beta_const3, beta_prpblck3, beta_income3 = model_results3.params
    print(f"\tln(psode) = {beta_str}0 + {beta_str}1*prpblck + {beta_str}2*ln(income) + u")
    print(f"\tln(psode) = {beta_const3:,.4f} + {beta_prpblck3:,.4f}*prpblck + {beta_income3:,.4f}*ln(income) + u")
    print(f"\tThe meaning of ß1 is the percentage change of psode with a change of 1 unit in prpblck.")
    print(f"\tThe meaning of ß2 is the percentage change of psode with a change of 1% in the income.")

    print_section('e')
    model_results4 = run_OLS(fast_food_df, 'psoda', ['prpblck', 'income', 'prppov'], add_constant=True, y_func=np.log,
                             x_func=[None, np.log, None])
    beta_const4, beta_prpblck4, beta_income4, beta_prppov4 = model_results4.params
    print(f"\tln(psode) = {beta_str}0 + {beta_str}1*prpblck + {beta_str}2*ln(income) + {beta_str}3*prppov + u")
    print(f"\tln(psode) = {beta_const4:,.4f} + {beta_prpblck4:,.4f}*prpblck + {beta_income4:,.4f}*ln(income) "
          f" +  {beta_prppov4:,.4f}*prppov + u")
    print(f"\tThe Model Results:\n{model_results4.summary2()}\n")
    print(f'\t{beta_str}_prpblck decreased when adding the prppov, as it explains the price better than the percentile '
          f'of black people.')
    print_section('f')
    correlation = np.log(fast_food_df['income']).corr(fast_food_df['prppov'])
    print(f'\tThe correlation between log(income) and prppov is {correlation:.4f}.')
    print(f'\tThe negative correlation is as high as I would expect. As suggested by the names of these features, the '
          f'\t\n income is high in richer neighborhoods - inverse correlation with poverty.')


if __name__ == '__main__':
    print('55812 Data analysis for decision making\nEx2\nLiat Meir (305677817) & Omer Arie Lerinman (304919863)\n\n')
    data = load_data()
    first_question(data)
    second_question(data)
    print('Done.')

