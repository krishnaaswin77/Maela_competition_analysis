import numpy as np
import pandas as pd
import datetime
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import to_long_format
from lifelines.utils import add_covariate_to_timeline
from lifelines.statistics import proportional_hazard_test
from lifelines import CoxTimeVaryingFitter
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import cycle
import random
from scipy.odr import ODR, Model, RealData


def sero_survival_basic(full_data):
    # QUESTION: Do serotypes affect the duration of carriage?
    # This is already established knowledge that some serotypes are longer carried than others.
    # We want to confi rm this here, and compare with literature about serotype carriage durations

    # Equation:
    # CARRIAGE DURATION = F( SEROTYPE IDENTITY )

    data_df = full_data[['agestrat', 'observed', 'durations']].reset_index(drop=1)

    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression
   
    # Simple survival analysis and getting Kaplan meier function
    kmf = KaplanMeierFitter()
    kmf.fit(data_df.durations, event_observed=data_df.observed)
    # kmf.survival_function_.plot()
    # plt.title('Survival function')

    events = data_df['observed']
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if data_df.loc[~events, sero].var() == 0]
    
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    data_df = data_df.loc[(data_df[removed_sero] != 1).all(axis=1)]
    data_df = data_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    # Cox proportional hazards regression on age and serotype identity
    cph = CoxPHFitter()
    cph.fit(data_df, duration_col='durations', event_col='observed',
            strata='agestrat')

    # cph.print_summary()
    # axis = plt.subplot()
    # cph.plot(ax=axis)

    return kmf, cph, data_df


def basic_survival_with_drugs(full_data, simple_drug_df):
    # QUESTION: Does taking drugs have an overall affect on the duration of carriage?
    # Ignoring resistance of strains, we model how simple drug use affects clearance rate

    # Equation:
    # CARRIAGE DURATION = F( DRUG-USE, SEROTYPE IDENTITY )
    
    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index(drop=True)
    drug_df = simple_drug_df.reset_index(drop=True)
    
    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, drug_df, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")
    req_df = req_df.dropna()
    
    events = req_df['observed']
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]
    
    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    cph_d = CoxTimeVaryingFitter(penalizer=0.1)
    cph_d.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
              strata='agestrat', show_progress=True)
    
    return cph_d


def cc_on_survival(full_data, cocarriage_events, interaction=0):
    # QUESTION: Does co-colonisation reduce the duration of carriage?
    # This is the hypothesis suggesting within-host competition between strains. We add age as covariate as frequency
    # of carriage changes with age. We co-vary serotype as this is a proxy for some strains being longer than others in
    # carriage, and longer carriages may have more overlap

    # QUESTION2: Can Serotype determine clearance priority under within-host competition?
    # When interaction=1, include interaction terms of competitor with serotypes
    
    # Equation:
    # CARRIAGE DURATION = F(SEROTYPE IDENTITY, CO-COLONIZATION STATE, [SERO-INTERACTION TERMS IF NEEDED] )

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index(drop=True)
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index(drop=True)
    
    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    removed_sero = dummies.columns[(dummies.sum() <= 50)]  # Only when interaction==1 to avoid convergence issues
    # removed_sero = removed_sero.append(pd.Index(['15A', '21', '35F']))
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")
    req_df = req_df.dropna()   # Nan values where a baby was sampled only once so no durations inferred.
    
    if interaction==0:
        events = req_df['observed']  # I remove serotypes which have low variance in observed clearance
        seroNames = dummies.columns.drop('18C')
        removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]
    
    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    # Renaming serotypes
    exclude_columns = ['Co_carried', 'agestrat', 'start', 'stop', 'Carriage_ID', 'observed']
    req_df.rename(columns=lambda x: f'sero_{x}' if x not in exclude_columns else x, inplace=True)
    
    equation = "+".join(req_df.columns.drop(["Carriage_ID", "start", "stop", "agestrat", "observed"]))
    if interaction == 1:
        sero_list = req_df.columns.drop(["Carriage_ID", "start", "stop", "Co_carried", "agestrat", "observed"])
        interaction_terms = "+".join([f"{col}:Co_carried" for col in sero_list])
        equation = equation + f"+{interaction_terms}"
    sur_cc = CoxTimeVaryingFitter(penalizer=0.01)
    sur_cc.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
               strata='agestrat', formula=equation, show_progress=True)

    return sur_cc


def cc_on_survival_independent(independent_full_data, cocarriage_events):
    # QUESTION: Does co-colonisation reduce the duration of carriage?
    # This is the same question as previous one with a difference in the input data. We know that when there is
    # co-carriage, i.e., overlap of strain carriage, the assumption of independence for regression would be lost.
    # Here, I remedy it by keeping only one co-carrying pair every time there is co-carriage = 1. This comes from
    # the function def - full_data_with_independence(full_data)

    # EQUATION:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CO-COLONIZATION STATE )
    sur_cc_ind = cc_on_survival(independent_full_data, cocarriage_events)

    return sur_cc_ind


def cc_resident_priority_effect(full_data, cocarriage_events):
    # QUESTION: Do competing strains that arrived first in host have priority in clearance?
    # We define COCARRIAGE-WITH-PRIORITY, COCARRIAGE-WITHOUT PRIORITY in terms of which of the competing strains
    # arrived first (priority) and which came later (no priority). Baseline should be in this case no co-carriage.

    # Equation:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, COCARRIAGE-WITH-PRIORITY, COCARRIAGE-WITHOUT PRIORITY,
    #                        PRIORITY-WITHOUT-COCARRIAGE)

    data_df = full_data[['Carriage_ID', 'agestrat', 'serotype_carried', 'observed', 'durations']].reset_index(drop=True)
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index(drop=True)
    co2 = co_data.loc[co_data['Time'] == 0][['Carriage_ID', 'Co_carried']].rename(columns={'Co_carried': 'second_coloniser'})
    # This command finds second_coloniser - 1 when the strain enters the host niche that is already occupied
    # To do this, I look at the value of Co_carried of each episode of carriage when the establishment just happened, i.e, Time=0
    data_df = data_df.merge(co2, on='Carriage_ID', how='left')

    dummies = pd.get_dummies(data_df['serotype_carried'])  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")
    req_df = req_df.dropna()   # Nan values where a baby was sampled only once so no durations inferred.

    req_df = req_df.assign(cc_priority=req_df.Co_carried*(1-req_df.second_coloniser),
                           cc_wo_priority=req_df.Co_carried*req_df.second_coloniser)
    
    # I remove serotypes which have low variance in observed clearance
    events = req_df['observed']
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)
    req_df = req_df.drop(['Co_carried', 'second_coloniser', 'serotype_carried'], axis=1)
    req_df = req_df.dropna()

    sur_pr = CoxTimeVaryingFitter(penalizer=0.01)
    sur_pr.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
               strata='agestrat', show_progress=False)

    return sur_pr


def permute_resident_effect_test(full_data, times=100):

    data_df = full_data[['Infant', 'Carriage_ID', 'carriage_start', 'serotype_carried',
                         'agestrat', 'observed', 'samp_start', 'samp_end', 'durations']]
    null_dist_cc_priority, null_dist_cc_wo_priority = [], []

    # FOR EACH PERMUTATION:
    for perm in range(times):
        grouped_data = data_df.groupby('Infant')
        randomized_data = grouped_data.apply(randomize_carriage).droplevel(0) # Droplevel to remove grouping
        randomized_cc_data = get_cc_data(randomized_data)
        rand_sur = cc_resident_priority_effect(randomized_data, randomized_cc_data)
        summary_df = rand_sur.summary
        null_dist_cc_priority.append(summary_df['coef']['cc_priority'])
        null_dist_cc_wo_priority.append(summary_df['coef']['cc_wo_priority'])

    return null_dist_cc_priority, null_dist_cc_wo_priority


def effects_of_drugs(full_data, cocarriage_events, effective_drug_df, ineffective_drug_df):
    # Question:
    # What is the effect of drugs on the clearance rate of resistant/sensitive strains?

    # Equation:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CC STATE,
    #                        EFFECTIVE DRUG USE, INEFFECTIVE DRUG USE)

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index(drop=True)
    drug_eff = effective_drug_df.copy()
    drug_ineff = ineffective_drug_df.copy()
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index(drop=True)

    dummies = pd.get_dummies(full_data['serotype_carried']).reset_index(
        drop=True)  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    # Adding three time-dependent variables - co-carriage, effective and ineffective drug usage
    req_df = data_df.pipe(add_covariate_to_timeline, drug_eff, duration_col="Time", id_col="Carriage_ID",
                          event_col="observed") \
        .pipe(add_covariate_to_timeline, drug_ineff, duration_col="Time", id_col="Carriage_ID", event_col="observed") \
        .pipe(add_covariate_to_timeline, co_data, duration_col="Time", id_col="Carriage_ID", event_col="observed")
    req_df = req_df.dropna()  # Also drops resistance-unannotated carriages

    events = req_df['observed']  # I remove serotypes which have low variance in observed clearance
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    ctv_benefit = CoxTimeVaryingFitter(penalizer=0.1)
    ctv_benefit.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
                    strata='agestrat',
                    show_progress=True)

    return ctv_benefit


def cc_tradeoff_with_avg_duration(full_data, cocarriage_events):
    # QUESTION: Do long-carried strains have a disadvantage for clearance under co-carriage?
    # I want to look for a trade-off between clearance rate and competitive ability within-host.
    # I look at the average single-carriage-duration-of-carriage of serotypes. Then I add this avgD
    # into the analysis and its interaction with co-carriage. I expect serotypes with long carriage
    # should face higher clearance rate under co-carriage.

    # EQUATION:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CO-COLONIZATION STATE x AVG.DURATION )

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations', 'avgD']].reset_index()
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index()

    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    removed_sero = dummies.columns[(dummies.sum() <= 5)]  # Low frequency serotypes to remove from regression
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")
    req_df = req_df.dropna()

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    equation = "+".join(req_df.columns.drop(["Carriage_ID", "start", "stop"])) + "+avgD*Co_carried"
    sur_cc = CoxTimeVaryingFitter(penalizer=0.1)
    sur_cc.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
               formula=equation, strata='agestrat',
               show_progress=True)

    return sur_cc


def sero_immunity_on_survival(full_data, cocarriage_events):
    # QUESTION: Does serotype-specific immunity affect clearance of serotypes?
    # Is there evidence for serotype-specific immunity in this data? For this we look at whether subsequent colonisations
    # by the same serotype are affected by the earlier colonisations of the same serotype. For each carriage, I will calculate
    # the number of colonisations by the same serotype before that carriage (= Sero-history), to include in the survival analysis.
    # With serotype-specific immunity, we should expect this to have a significant positive coefficient on clearance rate.
    # Similarly, you can also

    # EQUATION:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CO-COLONIZATION STATE, SERO-HISTORY/GEN-IMMUNITY )

    # d = datetime.timedelta(days=180)
    # full_data['gen_immunity1'] = full_data.groupby('Infant')['carriage_start'] \
    #     .transform(lambda group: np.array([((group > (x - d)) & (group < x)).sum() for x in group.values])) \
    #     .astype(int)
    # full_data['gen_immunity2'] = full_data.groupby('Infant')['carriage_start'] \
    #     .transform(lambda group: np.array([((group <= (x - d))).sum() for x in group.values])) \
    #     .astype(int)

    # You can choose to keep one or both of generic and sero-specific immunity here
    data_df = full_data[['Carriage_ID', 'age', 'observed', 'durations', 'gen_immunity1', 'sero_history1', 'gen_immunity2', 'sero_history2']].reset_index()
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index()

    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")
    req_df = req_df.dropna()

    # I remove serotypes which have low variance in observed clearance
    events = req_df['observed']
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)
    req_df = req_df.drop(['index'], axis=1)

    # Checked the gen_immunity*serotype interaction. This interaction term is always insignificant.
    sur_immune = CoxTimeVaryingFitter(penalizer=0.1)
    sur_immune.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
                   show_progress=True)

    return sur_immune


def cc_tradeoff_diff_avg_duration(full_data, cocarriage_events):
    # QUESTION: Does fitness difference between co-carrying strains affect which gets cleared?

    # Equation:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CO-COLONIZER-FITTER, CO-COLONIZER-WEAKER )
    # The co-colonizer fitness identity is its avgD difference from focal strain
    # (1) Fitter = Higher duration than focal strain
    # (2) Weaker = Lower duration than focal strain

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations', 'avgD']].reset_index()
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried', 'avgD_cc']].reset_index()

    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    removed_sero = dummies.columns[(dummies.sum() <= 10)]  # Low frequency serotypes to remove from regression
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression
    # Take only those serotypes into regression which have at least 5 carriage episodes in the
    # dataset.

    data_df = to_long_format(data_df, duration_col="durations")

    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")

    # Below, I create the CO-COLONIZER-FITTER and CO-COLONIZER-WEAKER columns
    req_df = req_df.assign(CC_Fit=lambda x: (x['Co_carried'] == 1) & (x['avgD_cc'] >= x['avgD']),
                           CC_Wkr=lambda x: (x['Co_carried'] == 1) & (x['avgD_cc'] < x['avgD']))
    req_df = req_df.assign(CC_Fit=req_df.CC_Fit.astype('int'),
                           CC_Wkr=req_df.CC_Wkr.astype('int'))
    req_df = req_df.drop('Co_carried', axis=1)
    req_df = req_df.drop(['avgD_cc', 'avgD'], axis=1)
    req_df = req_df.dropna()

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    ctv_cc = CoxTimeVaryingFitter(penalizer=0.1)
    ctv_cc.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
               strata='agestrat', show_progress=True)

    return ctv_cc


def focal_res_on_survival(full_data, cocarriage_events,
                          drug, interaction=1):
    # QUESTION: (1) Does resistance to a drug affect clearance rate of serotypes in the data?
    # (2) Does the resistance status of the focal strain determine its effect by co-colonisation (when interaction==1)?
    # That is, is a resistant focal strain affected more by a co-colonizer than a sensitive focal strain?

    # Equation:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CC STATE, RES OF FOCAL STRAIN, CC x RES OF FOCAL (If needed) )
    # Here, I would use the interaction term to find out whether the co-colonisation effect is enhanced or
    # decreased by whether the focal strain is resistant or not.
    # I can measure resistance in two ways;
    # (1) By looking at each drug-resistance separately
    # (2) By looking at all drug-resistances combined

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index(drop=True)
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index(drop=True)

    # --------------------
    # (1) Below I get the resistance status of the focal strain. Use one of the below resistances.
    # chloramphenicol, clindamycin, co.trimoxazole, erythromycin, penicillin, tetracycline
    # There are only two carriages with ceftriaxone, so I don't use this antibiotic
    if drug == 'Any':
        # Here I look at all the resistances combined
        data_df = data_df.assign(
            foc_res=full_data['resistance_profile'].reset_index(drop=True).apply(lambda x: x.max() if x.empty == 0 else np.nan))
    else:
        data_df = data_df.assign(foc_res=full_data['resistance_profile'].reset_index(drop=True).apply(lambda x: x[drug] if x.empty == 0 else np.nan))
    # --------------------

    # Adding serotypes as dummy variables
    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    # removed_sero = dummies.columns[(dummies.sum() <= 10)]  # Low frequency serotypes to remove from regression
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")
    req_df = req_df.dropna()

    events = req_df['observed']  # I remove serotypes which have low variance in observed clearance
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)] # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    req_df = req_df.assign(foc_res=req_df['foc_res'].astype('int'))
    
    # Renaming the serotypes
    exclude_columns = ['Co_carried', 'agestrat',
                        'foc_res', 'start', 'stop', 'Carriage_ID', 'observed']
    req_df.rename(columns=lambda x: f'sero_{x}' if x not in exclude_columns else x, inplace=True)

    equation = "+".join(req_df.columns.drop(["Carriage_ID", "start", "stop", "observed", "agestrat"]))
    if interaction == 1:
        equation = "+".join(req_df.columns.drop(["Carriage_ID", "start", "stop", "observed", "agestrat"])) + "+foc_res:Co_carried"
    ctv_fr = CoxTimeVaryingFitter(penalizer=0.1)
    ctv_fr.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
               formula=equation, strata='agestrat',
               show_progress=True)

    return ctv_fr


def cocol_res_on_survival(full_data, cocarriage_events, drug):
    # QUESTION: Does co-coloniser's resistance-state affect the rate of clearance under co-colonisation?

    # Equation:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CO-COLONIZER-RESISTANT, CO-COLONIZER-SENSITIVE)
    # The co-colonizer resistant identity can be done in two ways. Here I do (1).
    # (1) By looking at each drug-resistance separately
    # (2) By looking at any resistance

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index(drop=True)
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index(drop=True)

    # Below I add the resistance column
    if drug == 'Any':
        co_data = co_data.assign(
            Res=cocarriage_events['Co_resistance'].reset_index(drop=True).apply(lambda x: x.max() if x.empty == 0 else np.nan))
    else:
        co_data = co_data.assign(
            Res=cocarriage_events['Co_resistance'].reset_index(drop=True).apply(lambda x: x[drug] if x.empty == 0 else np.nan))

    dummies = pd.get_dummies(full_data['serotype_carried'])  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df. reset_index for index alignment
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    req_df = add_covariate_to_timeline(data_df, co_data, duration_col="Time", id_col="Carriage_ID",
                                       event_col="observed")

    # Below, I create the CC_with_resistance and CC_with sensitive columns
    req_df = req_df.assign(CC_Res=lambda x: (x['Co_carried'] == 1) & (x['Res'] == 1),
                           CC_Sen=lambda x: (x['Co_carried'] == 1) & (x['Res'] == 0))
    req_df = req_df.assign(CC_Res=req_df.CC_Res.astype('int'),
                           CC_Sen=req_df.CC_Sen.astype('int'))
    req_df = req_df.drop(['Res', 'Co_carried'], axis=1)
    req_df = req_df.dropna()

    events = req_df['observed']  # I remove serotypes which have low variance in observed clearance
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]
    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    ctv_rc = CoxTimeVaryingFitter(penalizer=0.1)
    ctv_rc.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
               strata='agestrat', show_progress=True)

    return ctv_rc


def cost_of_resistance(full_data, cocarriage_events, D_minus_df,
                       effective_drug_df, ineffective_drug_df, interaction=0):
    # Question:
    # How does resistance on the focal strain affect the clearance rate?
    # Here, I do the version with RES-WITHOUT-DRUG - correcting for treatment

    # Equation:
    # CARRIAGE DURATION = F( SEROTYPE IDENTITY, CC STATE, RES-WITHOUT-DRUG, CC x RES OF FOCAL,
    #                        EFFECTIVE DRUG USE, INEFFECTIVE DRUG USE,
    #                        CC x INEFF D.U.)

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index(drop=True)
    drug_eff = effective_drug_df.copy()
    drug_ineff = ineffective_drug_df.copy()
    D_min_df = D_minus_df.copy()
    # D_min_df = D_min_df.rename(columns={'D_minus': 'foc_res'})
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index(drop=True)

    dummies = pd.get_dummies(full_data['serotype_carried']).reset_index(drop=True)  # Make dummy variables for serotype carried
    data_df = data_df.join(dummies.reset_index(drop=True))  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    # Adding three time-dependent variables - co-carriage, effective and ineffective drug usage, and resistance
    req_df = data_df.pipe(add_covariate_to_timeline, drug_eff, duration_col="Time", id_col="Carriage_ID",
                          event_col="observed") \
        .pipe(add_covariate_to_timeline, drug_ineff, duration_col="Time", id_col="Carriage_ID", event_col="observed") \
        .pipe(add_covariate_to_timeline, D_min_df, duration_col="Time", id_col="Carriage_ID", event_col="observed") \
        .pipe(add_covariate_to_timeline, co_data, duration_col="Time", id_col="Carriage_ID", event_col="observed")
    req_df = req_df.dropna()  # Also drops resistance-unannotated carriages
    
    events = req_df['observed']  # I remove serotypes which have low variance in observed clearance
    seroNames = dummies.columns.drop('18C')
    removed_sero = [sero for sero in seroNames if req_df.loc[~events, sero].var() == 0]

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    # Renaming the serotypes (problems with interpreting some names)
    exclude_columns = ['Co_carried', 'D_minus', 'Drug_use', 'Ineff_Drug_use',
       'agestrat', 'start', 'stop', 'Carriage_ID', 'observed']
    req_df.rename(columns=lambda x: f'sero_{x}' if x not in exclude_columns else x, inplace=True)
    
    equation = "+".join(req_df.columns.drop(["Carriage_ID", "start", "stop", "agestrat", "observed"]))
    if interaction == 1:
        equation = equation + "+D_minus:Co_carried"
    ctv_focall = CoxTimeVaryingFitter(penalizer=0.1)
    ctv_focall.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
                   formula=equation, strata='agestrat',
                   show_progress=True)

    return ctv_focall


def cost_of_resistance_detailed(full_data, cocarriage_events, resistance_df, D_minus_df,
                                effective_drug_df, ineffective_drug_df):
    # Question: Does focal strain resistance status and co-colonizer resistance status together matter
    # for the effect of co-colonizer's increased clearance rate?
    # i.e., does a sensitive co-colonizer clear a resistant focal strain faster than a sensitive?

    # EQUATION:
    # CARRIAGE DURATION = F( AGE, SEROTYPE IDENTITY, CC STATE,

    data_df = full_data[['Carriage_ID', 'agestrat', 'observed', 'durations']].reset_index()
    drug_eff = effective_drug_df.copy()
    drug_ineff = ineffective_drug_df.copy()
    D_min_df = D_minus_df.copy()
    res_table = resistance_df[['Carriage_ID', 'D_plus']].reset_index()
    co_data = cocarriage_events[['Carriage_ID', 'Time', 'Co_carried']].reset_index()

    # Below I get the resistance status of the focal strain. D+ RESISTANCE ONLY.
    data_df = data_df.merge(res_table, how='outer', left_on='Carriage_ID', right_on='Carriage_ID')

    # Now for the co-colonizer's resistance status - I look at one of these
    # chloramphenicol, clindamycin, co.trimoxazole, erythromycin, penicillin, tetracycline
    co_data = co_data.assign(
        co_res=cocarriage_events['Co_resistance'].apply(lambda x: x['erythromycin'] if x.empty == 0 else np.nan))

    dummies = pd.get_dummies(full_data['serotype_carried']).reset_index(
        drop=True)  # Make dummy variables for serotype carried
    removed_sero = dummies.columns[(dummies.sum() <= 5)]  # Low frequency serotypes to remove from regression
    data_df = data_df.join(dummies)  # Add dummy columns to the data_df
    data_df = data_df.drop('18C', axis=1)  # this is the reference serotype - remove it from regression

    data_df = to_long_format(data_df, duration_col="durations")
    # Adding three time-dependent variables - co-carriage, effective and ineffective drug usage
    req_df = data_df.pipe(add_covariate_to_timeline, drug_eff, duration_col="Time", id_col="Carriage_ID",
                          event_col="observed") \
        .pipe(add_covariate_to_timeline, drug_ineff, duration_col="Time", id_col="Carriage_ID", event_col="observed") \
        .pipe(add_covariate_to_timeline, D_min_df, duration_col="Time", id_col="Carriage_ID", event_col="observed") \
        .pipe(add_covariate_to_timeline, co_data, duration_col="Time", id_col="Carriage_ID", event_col="observed")

    # Below, I create the CC_with_resistance and CC_with sensitive columns
    req_df = req_df.assign(CC_Res=lambda x: (x['Co_carried'] == 1) & (x['co_res'] == 1),
                           CC_Sen=lambda x: (x['Co_carried'] == 1) & (x['co_res'] == 0))
    req_df = req_df.assign(CC_Res=req_df.CC_Res.astype('int'),
                           CC_Sen=req_df.CC_Sen.astype('int'))
    req_df = req_df.drop(['Co_carried', 'co_res', 'D_plus'], axis=1)
    req_df = req_df.dropna()

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    req_df = req_df.assign(D_minus=req_df['D_minus'].astype('int'))

    equation = "+".join(req_df.columns.drop(["Carriage_ID", "start", "stop"])) \
               + "+D_minus*CC_Res+Ineff_Drug_use*CC_Res+D_minus*CC_Sen+Ineff_Drug_use*CC_Sen"
    ctv_foc_x_cc = CoxTimeVaryingFitter(penalizer=0.1)
    ctv_foc_x_cc.fit(req_df, id_col="Carriage_ID", event_col="observed", start_col="start", stop_col="stop",
                     formula=equation, strata='agestrat',
                     show_progress=True)

    return ctv_foc_x_cc


def linear_func(beta, x):
    return beta[0] * x + beta[1]


def run_odr(summary_df):
    all_cov = summary_df.index.drop(['Co_carried'])
    val_len = int(len(all_cov) / 2)

    covariates_sero = all_cov[:val_len]
    covariates_int = all_cov[val_len:]

    hazards_sero = summary_df.loc[covariates_sero]['coef']
    hazards_int = summary_df.loc[covariates_int]['coef']

    err_sero = summary_df.loc[covariates_sero]['se(coef)']
    err_int = summary_df.loc[covariates_int]['se(coef)']

    # Create a model and a RealData object
    linear_model = Model(linear_func)
    data = RealData(hazards_sero, hazards_int, sx=err_sero, sy=err_int)

    # Set up ODR with the model and data
    odr = ODR(data, linear_model, beta0=[1., 1.])
    output = odr.run()
    slope, intercept = output.beta

    return output


def plot_all_covariate_model(summary_df, covariates_to_plot, y_labels,
                             size, xlimits, filename):
    summary_df = summary_df[['coef', 'coef lower 95%', 'coef upper 95%']]  # Choose columns
    summary_df = summary_df.loc[covariates_to_plot]  # Get only these rows to plot

    fig = plt.figure(figsize=size)
    y_pos = np.arange(len(y_labels))
    # Add axes: [left, bottom, width, height]
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

    # Customizing error bars
    ax.errorbar(summary_df['coef'], y_pos,
                xerr=[(summary_df['coef'] - summary_df['coef lower 95%']),
                      (summary_df['coef upper 95%'] - summary_df['coef'])],
                fmt='o', color=[211/255, 86/255, 33/255], markersize=7, capsize=4, ecolor='navy', elinewidth=1.5)

    # Plotting a horizontal line at x=0 , color=[211/255, 86/255, 33/255]
    ax.axvline(x=0, ls='--', color='darkgrey', linewidth=3)

    # Adjusting tick positions and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12, color='black')
    [t.set_color('black') for t in ax.xaxis.get_ticklabels()]
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # Setting x-axis limits
    ax.set_xlim(xlimits[0], xlimits[1])
    ax.set_ylim(-0.5, len(y_labels) - 1 + 0.5)
    # ax.set_xscale('symlog')
    # Adding labels and title
    ax.set_xlabel('Hazard Coefficient in time-varying Cox Model', fontsize=12)
    
    # # Removing the top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # Adding grid lines
    ax.grid(axis='x', linestyle='--', color='grey')

    # plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    # plt.gca().xaxis.get_major_formatter().set_scientific(False)

    # Adjusting layout and padding
    # plt.tight_layout(pad=0.5)

    # Displaying the plot
    plt.savefig(filename, transparent=True)
    plt.show()

    return 0


def plot_sero_effects(summary_df,
                      size, filename):
    all_cov = summary_df.index.drop(['Co_carried'])
    val_len = int(len(all_cov) / 2)

    covariates_sero = all_cov[:val_len]
    covariates_int = all_cov[val_len:]

    hazards_sero = summary_df.loc[covariates_sero]['coef']
    hazards_int = summary_df.loc[covariates_int]['coef']
    
    err_sero = summary_df.loc[covariates_sero]['se(coef)']
    err_int = summary_df.loc[covariates_int]['se(coef)']

    # Create a model and a RealData object
    linear_model = Model(linear_func)
    data = RealData(hazards_sero, hazards_int, sx=err_sero, sy=err_int)

    # Set up ODR with the model and data
    odr = ODR(data, linear_model, beta0=[1., 1.])
    output = odr.run()
    slope, intercept = output.beta
    
    # Plot data points with error bars
    fig = plt.figure(figsize=size)

    # Add axes: [left, bottom, width, height]
    ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])

    ax.errorbar(hazards_sero, hazards_int, xerr=err_sero, yerr=err_int, fmt='o',
                color='mediumslateblue', capsize=2, label='Inferred hazards')

    # Plot the fitted line
    x_fit = np.linspace(min(hazards_sero), max(hazards_sero), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', label='Linear fit')
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.set_xlabel("Hazard coefficients of serotypes", fontsize=12)
    ax.set_ylabel("Hazard coefficients of\nserotype X presence-of-competitor", fontsize=12)
    ax.legend(loc="lower left")

    plt.savefig(filename, transparent=True)
    plt.show()
    
    return output


def plot_visualize_data(data_df, num=710, grid=True):
    # Using the datatable from OrganizeData, plot the serotype carriage visually for the first 100 serotypes.

    unique_infants = pd.unique(data_df['Infant'])
    unique_sero = pd.unique(data_df['serotype_carried'])

    get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
    sero_colors = dict(zip(unique_sero, get_colors(len(unique_sero))))  # create a dictionary of colours for each serotype
    person_number = 1

    fig, ax = plt.subplots(figsize=(20, 30))

    for infant in unique_infants:  # For each infant

        i = cycle([-0.1, 0, 0.1])  # separator to prevent overlapping lines
        infant_data = data_df.loc[data_df['Infant'] == infant]

        for index, carriage in infant_data.iterrows():  # Each carriage episode that needs to be plotted
            next_i = next(i)
            plt.plot([carriage['carriage_start'], carriage['carriage_end']], [person_number + next_i, person_number + next_i],
                     color=sero_colors[carriage['serotype_carried']], lw=3)

        person_number = person_number + 1
        if person_number > num:
            break

    ax.set_xlabel('Date', fontsize=30)
    ax.set_ylabel('Infants', fontsize=30)
    ax.set_xlim(min(data_df['carriage_start']), max(data_df.loc[:853]['carriage_end']))
    ax.set_ylim(0, num+1)
    ax.tick_params(axis='x', labelsize=26, rotation=45, length=10)
    ax.tick_params(axis='y', length=10)
    ax.set_yticks([i for i in range(num)], [''] * num)
    if grid:
        plt.grid(visible=True, which='both', linestyle='dotted')
    ax.spines['top'].set_linewidth(4)  # Set thickness for top border
    ax.spines['right'].set_linewidth(4)  # Set thickness for right border
    ax.spines['bottom'].set_linewidth(4)  # Set thickness for bottom border
    ax.spines['left'].set_linewidth(4)  # Set thickness for left border
    plt.savefig('figs1.svg')
    plt.show()
    return 0


########_________________________________________


def randomize_carriage(group):
    # This function takes the dataframe of episodes of carriage (sorted by infant)
    # Then it randomly shuffles the position of all episodes of carriage within each infant 
    # and thereby generates another possible permutation of carriages in each infant.
    
    # Extract samp_start and samp_end for the infant
    samp_start = group['samp_start'].min()
    samp_end = group['samp_end'].max()
    new_carriage = []

    # For each episode, randomly place the start time within the sampling window
    for _, row in group.iterrows():
        duration = row['durations']

        # Calculate the latest possible start time to ensure the episode fits within the sampling window
        latest_start = samp_end - pd.to_timedelta(duration, 'days')
        if latest_start > samp_start:
            # Randomize the new start time within the allowable window
            new_start = np.random.randint(int(samp_start.timestamp()), int(latest_start.timestamp()))
            new_start = pd.to_datetime(new_start, unit='s')
        else:           # In cases where only one episode of carriage sampled
            new_start = samp_start

        # Update carriage_start and carriage_end based on the new start time
        new_carriage.append({
            'Infant': row['Infant'],
            'Carriage_ID': row['Carriage_ID'],
            'carriage_start': new_start,
            'carriage_end': new_start + pd.to_timedelta(duration, 'days'),
            'serotype_carried': row['serotype_carried'],
            'age': (new_start - samp_start).days/30,
            'observed': row['observed'],
            'samp_start': samp_start,
            'samp_end': samp_end,
            'durations': duration
        })
    
    new_carriage = pd.DataFrame(new_carriage).reset_index(drop=True)
    new_carriage = new_carriage.assign(agestrat=pd.cut(new_carriage.age, np.arange(-0.01, 25, 12)))

    return pd.DataFrame(new_carriage)


def get_cc_data(randomized_data):
    # This function takes the dataframe of (randomized) episodes of carriage, sorted by infant
    # Then it regenerates the cocarriage_events_df dataframe for lifelines based on this new
    # randomized set of carriages. 
    # ALGORITHM: Take the set of all carriage start and end points of all carriages in an infant.
    # These are potential places where co-carriage status can change. Keep track of active carriages
    # as you move through time and record co-carriage status changes for all active carriages.
    
    # Initialize a list to store co-carriage events
    co_carriage_events = []

    # Iterate through each infant's data in the randomized dataset
    for infant, group in randomized_data.groupby('Infant'):
        # Sort the episodes by the start time
        group = group.sort_values('carriage_start')

        # Create a list to keep track of active carriages and their statuses
        active_carriages = []
        
        # Iterate through all possible start and end times (events)
        events = []
        for idx, row in group.iterrows():
            # Add start and end times as events
            events.append({'Time': row['carriage_start'], 'Carriage_ID': row['Carriage_ID'], 'Type': 'start'})
            events.append({'Time': row['carriage_end'], 'Carriage_ID': row['Carriage_ID'], 'Type': 'end'})

        # Sort events by time
        events = sorted(events, key=lambda x: x['Time'])
        
        # Process each event to update co-carriage status
        current_active_carriages = set()  # Track currently active episodes
        last_time = None
        
        for event in events:
            current_time = event['Time']
            
            # If time has changed, check the current state of co-carriage for all active carriages
            if last_time is not None and current_time != last_time:
                co_carried = 1 if len(current_active_carriages) > 1 else 0
                
                # Record the co-carriage status for all active carriages
                for carriage_id in current_active_carriages:
                    co_carriage_events.append({
                        'Carriage_ID': carriage_id,
                        'Time': last_time,
                        'Co_carried': co_carried
                    })

            # Update the list of active carriages based on the event type
            if event['Type'] == 'start':
                current_active_carriages.add(event['Carriage_ID'])
            elif event['Type'] == 'end':
                current_active_carriages.remove(event['Carriage_ID'])
            
            last_time = current_time
        
        # Final update for the last event in the timeline
        if last_time is not None:
            co_carried = 1 if len(current_active_carriages) > 1 else 0
            for carriage_id in current_active_carriages:
                co_carriage_events.append({
                    'Carriage_ID': carriage_id,
                    'Time': last_time,
                    'Co_carried': co_carried
                })

    co_carriage_events = pd.DataFrame(co_carriage_events).reset_index(drop=True)
    co_carriage_events = co_carriage_events.merge(randomized_data[['Carriage_ID', 'carriage_start']], on='Carriage_ID', how='left')
    co_carriage_events = co_carriage_events.assign(Time=pd.to_timedelta(co_carriage_events.Time - co_carriage_events.carriage_start).dt.days.astype('int'))

    return co_carriage_events


def plot_serotype_distribution_figures1(g_df):

    fig = plt.figure(figsize=(20, 8))

    # Add axes: [left, bottom, width, height]
    ax = fig.add_axes([0.1, 0.15, 0.88, 0.85])

    # Customizing error bars
    ax.bar(g_df.serotype_carried, g_df.durations/710, width=0.8,
           color=[1 / 255, 1 / 255, 1 / 255])
    plt.gca().set_xlim(-1, len(g_df.serotype_carried) -1)
    ax.tick_params(axis='x', labelsize=16, rotation=90, length=10)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_ylabel('Average time (in days) colonised\nby serotype per infant', fontsize=20)
    ax.set_xlabel('Serotype', fontsize=20)

    ax.spines['top'].set_linewidth(2)  # Set thickness for top border
    ax.spines['right'].set_linewidth(2)  # Set thickness for right border
    ax.spines['bottom'].set_linewidth(2)  # Set thickness for bottom border
    ax.spines['left'].set_linewidth(2)  # Set thickness for left border

    plt.savefig('figs1c.svg', transparent=True)
    plt.show()

    return 0
