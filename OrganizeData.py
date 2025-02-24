import numpy as np
import pandas as pd
import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def create_carriage_dataframe(ptype='Infant', int_time=0.5):
    # CREATES TWO OUTPUT DATAFRAMES
    # 1. CARRIAGE DATA = Dataframe where every row is a carriage and its corresponding properties
    # 2. COCARRIAGE EVENTS = Dataframe of points when co-carriage value changes between 1 and 0. This is in a format required by
    #                        the time-varying Cox model.
    # 3. SPEC_TRACKER = It's a dictionary that maps each carriage ID to all the specnums in that episode of carriage

    data_df = pd.DataFrame(
        columns=['Infant', 'Carriage_ID', 'carriage_start', 'carriage_end', 'serotype_carried', 'age', 'observed',
                 'samp_start', 'samp_end', 'resistance_profile'],
        index=[0])
    cocarriage_events = pd.DataFrame(
        columns=['Carriage_ID', 'Time', 'Co_carried', 'OS1', 'OS2', 'OS3', 'Co_resistance'],
        index=[0])
    #                                OS stands for other serotypes being co-carried

    data_raw = import_raw_data()
    # trans_raw = import_gerry_transmission_data(data_raw)

    specnum_to_serotype = data_raw[['specnum', 'serotype1', 'serotype2', 'serotype3', 'serotype4']].set_index(
        'specnum').apply(list, axis=1).to_dict()
    # Dictionary mapping specnum to serotype

    unique_infants = pd.unique(data_raw['codenum'])  # All unique infants

    unique_sero = pd.unique(data_raw['serotype1'])  # Get an array of all the unique serotypes
    unique_sero = unique_sero[1:]  # Removing the nan element
    unique_sero = np.delete(unique_sero, 2)  # Removing NT - non-typeable
    specnum_tracker = {}    # A dictionary that maps carriage ID to specnum

    carriage_index = 1  # A unique number that I will give to each carriage episode
    # I add the carriage_index to an episode when it starts (establishes)

    for infant in unique_infants:
        infant_data = data_raw.loc[(data_raw['codenum'] == infant) & (data_raw['category'] == ptype)]
        infant_data = infant_data.reset_index(drop=True)  # Reset new index 0,1,2,3 ....
        if infant_data.empty:  # If no infant data
            continue

        infant_start = infant_data['specdate'].iloc[0]
        infant_end = infant_data['specdate'].iloc[-1]

        sero_checker = {}  # Dictionary that contains carrying-serotype, ID and associated start-date
        specnum_tracker_t = {}  # Temporarily maps carriage ID to list of its specnums (sampling IDs)

        prev_checker = []  # Keeps track of the carrying serotypes from the previous time-step

        for index, person in infant_data.iterrows():
            curr_serotypes = [x for x in unique_sero if x in person[4:8].tolist()]
            lost_serotypes = [x for x in prev_checker if x not in curr_serotypes]
            gained_serotypes = [x for x in curr_serotypes if x not in prev_checker]

            for serotype in lost_serotypes:  # serotype is lost
                specnum_tracker.update({sero_checker[serotype][1]: specnum_tracker_t[sero_checker[serotype][1]]})
                end_delta_time = person[2] - infant_data['specdate'][index - 1]
                end_time = person[2] - end_delta_time * (1-int_time)

                start_time = sero_checker[serotype][0]
                age = end_time - infant_start  # age of infant when the serotype is lost

                res_prof = find_resistance_of_this_carriage(infant_data, start_time, end_time, serotype)

                new_row = pd.DataFrame(
                    [person[0], sero_checker[serotype][1], start_time, end_time, serotype, age, True, infant_start,
                     infant_end, res_prof]).T
                new_row.columns = ['Infant', 'Carriage_ID', 'carriage_start', 'carriage_end', 'serotype_carried', 'age',
                                   'observed', 'samp_start', 'samp_end', 'resistance_profile']
                data_df = pd.concat([data_df, new_row], ignore_index=True)

                del specnum_tracker_t[sero_checker[serotype][1]]
                del sero_checker[serotype]

            for serotype in gained_serotypes:  # if serotype is gained
                try:
                    start_delta_time = person[2] - infant_data['specdate'][index - 1]
                except:
                    start_delta_time = person[2] - person[2]  # this is just delta=0 in timestamp format

                start_time = person[2] - start_delta_time * int_time
                sero_checker.update(
                    {serotype: [start_time, carriage_index]})  # If new, add serotype, start date, #transmitted=0
                specnum_tracker_t.update({carriage_index: []})
                carriage_index = carriage_index + 1

            # At every time point, go through current serotypes and update if any transmissions happened for that serotype
            # Problem: 1 specnum can have multiple serotypes - so I have to check the receiving serotype and see if it matches with the transmitted serotype
            # for serotype in curr_serotypes:
            #     number_of_transmissions = get_num_transmissions(trans_raw, serotype, person[3], specnum_to_serotype)
            #     sero_checker[serotype][2] = sero_checker[serotype][2] + number_of_transmissions

            for carriage_id in specnum_tracker_t:
                specnum_tracker_t[carriage_id].append(person[3])

            # For new events that changed co-col state, update the co-colonisation dataframe
            if len(lost_serotypes) >= 1 or len(gained_serotypes) >= 1:
                co_colonized = int(len(sero_checker) > 1)  # if co-colonized then 1, else 0

                for serotype in gained_serotypes:
                    # update co-carriage status of newly gained serotypes
                    other_serotypes = list(sero_checker)
                    other_serotypes.remove(serotype)  # Other serotypes that are being co-carried
                    identity1, identity2, identity3 = other_serotypes + [0] * (3 - len(other_serotypes))
                    # All im trying to do is put 0 when there's nothing in identity 2

                    co_resistant = get_co_resistance(infant_data, other_serotypes, sero_checker)

                    new_co_row = pd.DataFrame([int(sero_checker[serotype][1]),
                                               sero_checker[serotype][0] - sero_checker[serotype][0],
                                               int(co_colonized),
                                               identity1,
                                               identity2,
                                               identity3,
                                               co_resistant]).T
                    new_co_row.columns = ['Carriage_ID', 'Time', 'Co_carried', 'OS1', 'OS2', 'OS3', 'Co_resistance']
                    cocarriage_events = pd.concat([cocarriage_events, new_co_row], ignore_index=True)

                for serotype in (set(sero_checker) - set(gained_serotypes)):
                    # update co-carriage status of non-gained serotypes. We don't have to care about lost serotypes
                    other_serotypes = list(sero_checker)
                    other_serotypes.remove(serotype)  # Other serotypes that are being co-carried
                    identity1, identity2, identity3 = other_serotypes + [0] * (3 - len(other_serotypes))
                    # All im trying to do is put 0 when there's nothing in identity 2/3

                    co_resistant = get_co_resistance(infant_data, other_serotypes, sero_checker)

                    lost_time = person[2] - (person[2] - infant_data['specdate'][index - 1]) / 2
                    new_co_row = pd.DataFrame([int(sero_checker[serotype][1]),
                                               lost_time - sero_checker[serotype][0],
                                               int(co_colonized),
                                               identity1,
                                               identity2,
                                               identity3,
                                               co_resistant]).T
                    new_co_row.columns = ['Carriage_ID', 'Time', 'Co_carried', 'OS1', 'OS2', 'OS3', 'Co_resistance']
                    cocarriage_events = pd.concat([cocarriage_events, new_co_row], ignore_index=True)

            prev_checker = curr_serotypes

        for serotype in sero_checker:  # Those serotypes whose clearances are not observed
            specnum_tracker.update(specnum_tracker_t)
            end_time = infant_end
            start_time = sero_checker[serotype][0]
            age = infant_end - infant_start

            res_prof = find_resistance_of_this_carriage(infant_data, start_time, end_time, serotype)

            new_row = pd.DataFrame(
                [infant_data['codenum'].iloc[0], sero_checker[serotype][1], start_time, end_time, serotype, age, False,
                 infant_start, infant_end,
                 res_prof]).T
            new_row.columns = ['Infant', 'Carriage_ID', 'carriage_start', 'carriage_end', 'serotype_carried', 'age',
                               'observed', 'samp_start', 'samp_end', 'resistance_profile']
            data_df = pd.concat([data_df, new_row], ignore_index=True)

            carriage_index = carriage_index + 1

    data_df.drop(index=data_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    cocarriage_events.drop(index=cocarriage_events.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    cocarriage_events = cocarriage_events.assign(Time=pd.to_timedelta(cocarriage_events['Time']).dt.days.astype('int'))  # Time from start when event happens, in days

    data_df = data_df.assign(observed=data_df.observed.astype('bool'))  # Data observed = 1, clearance observed
    data_df = data_df.assign(Carriage_ID=data_df.Carriage_ID.astype('int'))
    data_df = data_df.assign(age=pd.to_timedelta(data_df['age']).dt.days.astype('int') / 30)  # Age of clearance in months
    data_df = data_df.assign(durations=pd.to_timedelta(data_df['carriage_end'] - data_df['carriage_start']).dt.days.astype('int'))  # Carriage durations in days
    data_df = data_df.assign(category=ptype)
    data_df = data_df.assign(age=data_df.age - data_df.durations/30) # Age at colonization
    data_df = data_df.assign(agestrat=pd.cut(data_df.age, np.arange(-0.01, 25, 6)))
    
    # Add generic and serotype specific immunity
    data_df = data_df.assign(gen_immunity=data_df.groupby('Infant')['carriage_start']
                             .transform(lambda group: np.searchsorted(group.values, group.values, side='left')).astype(int))
        # Count number of carriages < current carriage_start
        # side='left' ensures that ties are counted correctly by excluding the current row from the count.

    data_df = data_df.assign(sero_history=data_df.groupby(['Infant', 'serotype_carried']).cumcount())

    return data_df, cocarriage_events, specnum_tracker


def full_data_with_independence(full_data):

    # The problem of independence: When strains are co-carried in the same infant, the assumption of independence of
    # samples in a statistical test might not hold, since one strain will influence the carriage of other.
    # Here, we try to solve to problem by considering leaving out these dependent carriages: when there are two
    # serotypes co-carried, we only keep one of them in the data, thereby removing the dependency between carriages.
    # This function outputs an alternate to carriage_df that is reduced to keep independence.

    independent_full_data = pd.DataFrame(columns=full_data.columns, index=[0])
    unique_infants = pd.unique(full_data['Infant'])

    for infant in unique_infants:
        sub_data = full_data.loc[full_data['Infant'] == infant]
        sub_data = sub_data.sort_values(by='carriage_start')
        prev_ending = full_data['carriage_start'][1] - datetime.timedelta(
            days=8000)  # This is like negative infinity time

        for index, episode in sub_data.iterrows():
            if episode['carriage_start'] > prev_ending:  # Greedy algorithm
                independent_full_data.loc[len(independent_full_data)] = episode.T  # Add this row
                prev_ending = episode['carriage_end']

    independent_full_data.drop(index=independent_full_data.index[0], axis=0,
                               inplace=True)  # Getting rid of first Nan row

    return independent_full_data


def create_carriage_df_with_moms(full_data, cocarriage_events):

    mother_data, cc_moms, _ = create_carriage_dataframe(ptype='Mother')
    mother_data = mother_data.assign(Carriage_ID=mother_data.Carriage_ID + 6017)  # Carriage IDs start after Infant IDs
    mother_data = mother_data.assign(age=48)  # Assume age of mothers doesn't matter, and age is fixed relative to infants at 4 years
    mother_data = mother_data.assign(category='Mother')

    cc_moms = cc_moms.assign(Carriage_ID=cc_moms.Carriage_ID + 6017)

    full_data_with_moms = pd.concat([full_data, mother_data], ignore_index=True)
    cc_with_moms = pd.concat([cocarriage_events, cc_moms], ignore_index=True)

    return full_data_with_moms, cc_with_moms


def find_resistance_of_this_carriage(infant_df, start, end, serotype):
    res_rows = infant_df.loc[(infant_df['specdate'] > start) & (infant_df['specdate'] < end)
                             & (infant_df['serotype'] == serotype)]
    if res_rows.empty:
        return res_rows.iloc[:, 8:15]

    res_rows = res_rows.iloc[:, 8:15]  # Columns with resistance profiles
    res_rows = res_rows.dropna()  # Remove all na rows

    resistance_profile = res_rows.mean()
    resistance_profile = resistance_profile.round()
    # The values are 0/1. But take the column average and round it off in case there are any errors
    # in one row that might cause issues

    return resistance_profile


def get_co_resistance(infant_df, other_serotypes, sero_checker):
    if len(other_serotypes) == 0:
        return pd.Series({'ceftriaxone': 0, 'chloramphenicol': 0,
                          'clindamycin': 0, 'co.trimoxazole': 0, 'erythromycin': 0,
                          'penicillin': 0, 'tetracycline': 0},
                         dtype='O')  # A zero-DF. When no co-carriage, this doesn't matter

    co_resistant_list = None
    other_checker = {key: sero_checker[key] for key in other_serotypes}
    for serotype in other_checker:
        start = other_checker[serotype][0]
        res_rows = infant_df.loc[(infant_df['specdate'] >= start) & (infant_df['serotype'] == serotype)]
        if res_rows.empty:  # If there is no resistance data for this serotype
            continue
        if co_resistant_list is None:
            co_resistant_list = res_rows.iloc[0, 8:15]
        else:
            co_resistant_list = np.maximum(co_resistant_list, res_rows.iloc[0, 8:15])
            # Assuming - first row here is the correct one and with no errors in resistance measurement
            # I do np.max because if one is res and other sens, I count total as resistant

    if co_resistant_list is None:
        co_resistant_list = res_rows.iloc[:, 8:15]

    return co_resistant_list


def import_raw_data():
    # Gets required data columns from the raw data files - specifically I want serotype carriages and resistance
    # carriages

    swab_summary = pd.read_csv('epi/Pneumo-DEEP_swab_summary.csv')
    isolate_summary = pd.read_csv('epi/Pneumo-DEEP_pnc_isolate_summary.csv')

    swab_details = swab_summary[['codenum', 'category', 'specdate', 'specnum',
                                 'serotype1', 'serotype2', 'serotype3', 'serotype4']]

    iso_details = isolate_summary[['specnum', 'ceftriaxone', 'chloramphenicol',
                                   'clindamycin', 'co.trimoxazole', 'erythromycin',
                                   'penicillin', 'tetracycline', 'serotype']]

    output_df = swab_details.merge(iso_details, how='outer', left_on='specnum', right_on='specnum')
    # Merge the two dataframes by their unique sampling code
    output_df = output_df.assign(
        specdate=pd.to_datetime(output_df['specdate']))  # Converts the time to time-readable format

    output_df = output_df.sort_values(by=['codenum', 'specdate'])  # Sort data by persons according to date

    output_df.iloc[:, 4:8] = output_df.iloc[:, 4:8].replace(to_replace='15B/C', value='15B_C')
    output_df.iloc[:, 4:8] = output_df.iloc[:, 4:8].replace(to_replace='6A/C', value='6A_C')

    output_df = output_df.replace(to_replace='SUSCEPTIBLE', value=0)
    output_df = output_df.replace(to_replace='RESISTANT', value=1)
    output_df = output_df.replace(to_replace='INTERMEDIATE', value=1)

    return output_df


def import_illness_data():
    # Get information about when patients got ill and took antibiotics and at which times

    illness_summary = pd.read_csv('epi/Pneumo-DEEP_illness_abx_summary.csv')
    illness_df = illness_summary[['codenum', 'date_start', 'drug1', 'drug2', 'drug3', 'drug4']]

    # From swab_summary, we can get details of which specnum relates to mother or infant.
    # So we can use nps-specnum column in the illness_summary to check whether the patients who got
    # antibiotic treatments are infants or mothers. I checked this and saw that all drugs were given to
    # infants, so I don't add this additional column to the df and just take all as infants.

    illness_df = illness_df.assign(date_start=pd.to_datetime(illness_df['date_start']))
    illness_df = illness_df.dropna(subset=['drug1'])  # Drop rows where drugs are not taken
    illness_df = illness_df.reset_index(drop=True)

    return illness_df


def effective_drug_plus_res(data_df, illness_df, treat_length=7):
    # Drug consumption as a time-varying covariate
    # It is a table that is of the format required by lifelines time-covariate regression
    # The drug column is yes/no : 0/1 valued. It says whether the carriage is associated with an
    # effective drug consumption or not. EFFECTIVE DRUG CONSUMPTION means a drug is consumed for which
    # the carriage does not confer any resistance.
    # In each carriage, I also look at the resistance carriage status. This is of two types: D+ and D-.
    # D+ Resistance means resistance is carried when the corresponding drug is also consumed during that carriage
    # episode. D- means resistance for which the corresponding drug is not taken.
    # This, I send as a separate dataframe, that is NOT time-co-varying (fixed per carriage).

    # ALGORITHM FOR DRUG:
    # I go through each illness episode (each row of illness_df) and check whether that illness episode (i.e. drug usage)
    # coincides with any carriage episode. There are three cases: (A) The illness episode is completely within
    # a carriage episode, (B) The illness episodes starts in the middle of a carriage episode, (C) The illness
    # episode ends in the middle of a carriage episode. For each case, the fraction of the carriage episode where
    # there is illness is marked with 1 for the drug column, and otherwise 0.
    effective_drug_df = pd.DataFrame(columns=['Carriage_ID', 'Time', 'Drug_use'],
                                     index=[0])
    unique_drugs = pd.concat([illness_df['drug1'], illness_df['drug2'].dropna()]).unique()  # All drugs prescribed

    # illness_df = illness_df.loc[(illness_df.drug1 == drug) | (illness_df.drug2 == drug) |
    #                             (illness_df.drug3 == drug) | (illness_df.drug4 == drug)]
    # Map between drugs and measured resistances
    # DRUG_DICT = { DRUG : CORRESPONDING RESISTANCE }
    drug_dict = {'AMOX/AMPICILLIN': 'penicillin',
                 'CIPROFLOXACIN': 'other',
                 'GENTAMICIN': 'other',
                 'ERYTHROMYCIN': 'erythromycin',
                 'CLOXACILLIN': 'penicillin', 
                 'CEFTRIAXONE': 'ceftriaxone',
                 'CLINDAMYCIN': 'clindamycin',
                 'CEFOTAXIME': 'ceftriaxone',
                 'TB TREATMENT': 'other',
                 'COTRIMOXAZOLE': 'co.trimoxazole',
                 'CEFIXIME': 'ceftriaxone',
                 'PENICILLIN V': 'penicillin',
                 'NITROFURANTOIN': 'other'
                 }
    # Some drugs have no corresponding resistance (at least in the given data). These are set to 'other'.
    # If one of these 'other' drugs are consumed, they are always effective.

    for index, drug_instance in illness_df.iterrows():   # Going through every illness episode

        patient = drug_instance['codenum']
        current_drugs = [x for x in unique_drugs if x in drug_instance[2:6].to_list()]  # Drugs under consumption
        # current_drugs = [drug]
        drug_resist = [drug_dict[k] for k in current_drugs]  # Drugs converted to resistance format
        drug_start = drug_instance['date_start']
        drug_end = drug_start + datetime.timedelta(days=treat_length)  # Drug effect in host lasts these many days

        # 1. When drug is taken during a colonisation event
        # drug_start date is sandwiched within the carriage start and end dates
        drug_start_rows = data_df[(data_df['Infant'] == patient) &
                                  (data_df['carriage_start'] <= drug_start) &
                                  (data_df['carriage_end'] >= drug_start)]  # relevant rows in this category
        for p, row in drug_start_rows.iterrows():

            # check whether drug is effective. If not, then continue to the next loop.
            resistances_carried = [x for x in row['resistance_profile'].index.to_list()
                                   if row['resistance_profile'][x] == 1]
            if len(set(drug_resist) - set(resistances_carried)) == 0:
                continue

            # Add row with carriage start drug use = 0
            new_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 0]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

            # add row with drug start
            new_row = pd.DataFrame([row['Carriage_ID'], drug_start - row['carriage_start'], 1]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

            # add row with drug end (if ending before cleared)
            if drug_end < row['carriage_end']:
                new_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 0]).T
                new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
                effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

            # Add resistance data
            # is_D_minus = bool(len(set(resistances_carried) - set(drug_resist)))
            # is_D_plus = bool(len(np.intersect1d(drug_resist, resistances_carried)))
            # res_row = pd.DataFrame([row['Carriage_ID'], is_D_minus, is_D_plus]).T
            # res_row.columns = ['Carriage_ID', 'D_minus', 'D_plus']
            # resistance_df = pd.concat([resistance_df, res_row], ignore_index=True)

        # 2. When drug effect is lost (= drug end) during colonisation event
        # drug_end is during colonisation but drug_start is before carriage starts - this is to avoid repeats
        drug_end_rows = data_df[(data_df['Infant'] == patient) &
                                (data_df['carriage_start'] <= drug_end) &
                                (data_df['carriage_end'] >= drug_end) & # Maybe i don't need this line
                                (data_df['carriage_start'] >= drug_start)]
        for p, row in drug_end_rows.iterrows():
            # check whether drug is effective (no resistance)
            resistances_carried = [x for x in row['resistance_profile'].index.to_list()
                                   if row['resistance_profile'][x] == 1]
            if len(set(drug_resist) - set(resistances_carried)) == 0:
                continue

            # add row with carriage start drug use = 1
            new_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 1]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

            # add row with drug end value changes to 0
            new_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 0]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

            # Add resistance data
            # is_D_minus = bool(len(set(resistances_carried) - set(drug_resist)))
            # is_D_plus = bool(len(np.intersect1d(drug_resist, resistances_carried)))
            # res_row = pd.DataFrame([row['Carriage_ID'], is_D_minus, is_D_plus]).T
            # res_row.columns = ['Carriage_ID', 'D_minus', 'D_plus']
            # resistance_df = pd.concat([resistance_df, res_row], ignore_index=True)

    effective_drug_df.drop(index=effective_drug_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    # resistance_df.drop(index=resistance_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row

    # 3. For all other carriage_IDs not covered, the value of Drug_use co-variate is set to 0
    remaining_ids = list(set(data_df['Carriage_ID']) - set(effective_drug_df['Carriage_ID']))
    rem_df = pd.DataFrame({'Carriage_ID': remaining_ids, 'Time': datetime.timedelta(0), 'Drug_use': 0})
    effective_drug_df = pd.concat([effective_drug_df, rem_df], ignore_index=True)

    # Dealing with overlaps and duplicates in effective_drug_df
    # Pure duplicates are removed and overlaps are set with the higher value 0 or 1 where it exists
    effective_drug_df = effective_drug_df.assign(
        Time=pd.to_timedelta(effective_drug_df['Time']).dt.days.astype('int'),
        Drug_use=effective_drug_df['Drug_use'].astype('int'))  # Time when drug intake starts, in days
    effective_drug_df = effective_drug_df.loc[effective_drug_df.groupby(['Carriage_ID', 'Time'])['Drug_use'].idxmax()]

    # Filling the resistance table - remaining
    # subset = data_df[data_df['Carriage_ID'].isin(remaining_ids)]['resistance_profile']
    # res_list = [s.sum() for s in subset]
    # D_minus_list = [bool(x) if isinstance(x, float) else np.nan for x in
    #                 res_list]  # Where resistance is not measured, put nan
    # D_plus_list = [0 if isinstance(x, float) else np.nan for x in res_list]
    # rem_res = pd.DataFrame({'Carriage_ID': remaining_ids, 'D_minus': D_minus_list,
    #                         'D_plus': D_plus_list})
    # resistance_df = pd.concat([resistance_df, rem_res], ignore_index=True)
    # resistance_df = resistance_df.drop_duplicates(subset=['Carriage_ID'], keep='first')

    return effective_drug_df


def get_simple_drug_df(data_df, illness_df, treat_length=7):
    # This is for getting when the hosts consume drugs, irrespective of resistance status.
    # This is in the format required for lifelines package for survival analysis.
    
    simple_drug_df = pd.DataFrame(columns=['Carriage_ID', 'Time', 'Drug_use'],
                                       index=[0])
    unique_drugs = pd.concat([illness_df['drug1'], illness_df['drug2'].dropna()]).unique()  # All drugs prescribed
    
    for index, drug_instance in illness_df.iterrows():

        patient = drug_instance['codenum']
        # current_drugs = [x for x in unique_drugs if x in drug_instance[2:6].to_list()]  # Drugs under consumption
        # current_drugs = [x for x in current_drugs if x is not None]  # removing None elements
        drug_start = drug_instance['date_start']
        drug_end = drug_start + datetime.timedelta(days=treat_length)  # Drug effect in host lasts 7 days
    
        # 1. When drug is taken during a colonisation event
        # drug_start date is sandwiched within the carriage start and end dates
        drug_start_rows = data_df[(data_df['Infant'] == patient) &
                                  (data_df['carriage_start'] <= drug_start) &
                                  (data_df['carriage_end'] >= drug_start)]  # relevant rows in this category

        for p, row in drug_start_rows.iterrows():

            # Add row with carriage start: drug use = 0
            new_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 0]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            simple_drug_df = pd.concat([simple_drug_df, new_row], ignore_index=True)

            # add row with drug start: drug use = 1
            new_row = pd.DataFrame([row['Carriage_ID'], drug_start - row['carriage_start'], 1]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            simple_drug_df = pd.concat([simple_drug_df, new_row], ignore_index=True)

            # add row if drug use ends before clearance
            if drug_end < row['carriage_end']:
                new_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 0]).T
                new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
                simple_drug_df = pd.concat([simple_drug_df, new_row], ignore_index=True)
                
        # 2. When drug effect is lost (= drug end) during colonisation event
        # drug_end is during colonisation but drug_start is before carriage starts - this is to avoid repeats
        drug_end_rows = data_df[(data_df['Infant'] == patient) &
                                (data_df['carriage_start'] <= drug_end) &
                                (data_df['carriage_end'] >= drug_end) &
                                (data_df['carriage_start'] >= drug_start)]

        for p, row in drug_end_rows.iterrows():
            # add row with carriage start drug use = 1
            new_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 1]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            simple_drug_df = pd.concat([simple_drug_df, new_row], ignore_index=True)

            # add row with drug end value changes to 0
            new_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 0]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Drug_use']
            simple_drug_df = pd.concat([simple_drug_df, new_row], ignore_index=True)
            
    simple_drug_df.drop(index=simple_drug_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    
    # 3. For all other carriage_IDs not covered, the value of Drug_use co-variate is set to 0
    remaining_ids = list(set(data_df['Carriage_ID']) - set(simple_drug_df['Carriage_ID']))
    rem_df = pd.DataFrame({'Carriage_ID': remaining_ids, 'Time': datetime.timedelta(0), 'Drug_use': 0})
    simple_drug_df = pd.concat([simple_drug_df, rem_df], ignore_index=True)
    simple_drug_df = simple_drug_df.assign(Time=pd.to_timedelta(simple_drug_df['Time']).dt.days.astype('int'), # Convert into integer format in days
                                           Drug_use=simple_drug_df.Drug_use.astype('int'))
    
    # Dealing with duplicates and overlaps: Keep 1 when duplicates have 0 and 1
    simple_drug_df = simple_drug_df.loc[simple_drug_df.groupby(['Carriage_ID', 'Time'])['Drug_use'].idxmax()]
    
    return simple_drug_df
    

def ineffective_drugs(data_df, illness_df, treat_length=7):
    # INEFFECTIVE DRUGS - Corresponds to those cases where drug is consumed but the corresponding resistance for that
    # drug is also carried.

    ineffective_drug_df = pd.DataFrame(columns=['Carriage_ID', 'Time', 'Ineff_Drug_use'],
                                       index=[0])
    D_min_res_df = pd.DataFrame(columns=['Carriage_ID', 'Time', 'D_minus'],
                                       index=[0])
    unique_drugs = pd.concat([illness_df['drug1'], illness_df['drug2'].dropna()]).unique()  # All drugs prescribed
    # illness_df = illness_df.loc[(illness_df.drug1==drug) | (illness_df.drug2==drug) |
    #                             (illness_df.drug3==drug) | (illness_df.drug4==drug)]

    # Map between drugs and measured resistances
    # DRUG_DICT = { DRUG : CORRESPONDING RESISTANCE }
    drug_dict = {'AMOX/AMPICILLIN': 'penicillin',
                 'CIPROFLOXACIN': 'other',
                 'GENTAMICIN': 'other',
                 'ERYTHROMYCIN': 'erythromycin',
                 'CLOXACILLIN': 'penicillin',
                 'CEFTRIAXONE': 'ceftriaxone',
                 'CLINDAMYCIN': 'clindamycin',
                 'CEFOTAXIME': 'ceftriaxone',
                 'TB TREATMENT': 'other',
                 'COTRIMOXAZOLE': 'co.trimoxazole',
                 'CEFIXIME': 'ceftriaxone',
                 'PENICILLIN V': 'penicillin',
                 'NITROFURANTOIN': 'other'
                 }

    for index, drug_instance in illness_df.iterrows():

        patient = drug_instance['codenum']
        current_drugs = [x for x in unique_drugs if x in drug_instance[2:6].to_list()]  # Drugs under consumption
        # current_drugs = [drug]
        drug_resist = [drug_dict[k] for k in current_drugs]  # Drugs taken in resistance-format
        drug_resist = [x for x in drug_resist if x is not None]  # removing None elements
        drug_start = drug_instance['date_start']
        drug_end = drug_start + datetime.timedelta(days=treat_length)  # Drug effect in host lasts 7 days

        # 1. When drug is taken during a colonisation event
        # drug_start date is sandwiched within the carriage start and end dates
        drug_start_rows = data_df[(data_df['Infant'] == patient) &
                                  (data_df['carriage_start'] <= drug_start) &
                                  (data_df['carriage_end'] >= drug_start)]  # relevant rows in this category

        for p, row in drug_start_rows.iterrows():

            # check whether drug is ineffective (has corresponding resistance)
            resistances_carried = [x for x in row['resistance_profile'].index.to_list()
                                   if row['resistance_profile'][x] == 1]
            if len(np.intersect1d(drug_resist, resistances_carried)) == 0:
                continue

            # Add row with carriage start: drug use = 0 and D_minus = 1
            new_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 0]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Ineff_Drug_use']
            ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
            new_res_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 1]).T
            new_res_row.columns = ['Carriage_ID', 'Time', 'D_minus']
            D_min_res_df = pd.concat([D_min_res_df, new_res_row], ignore_index=True)

            # add row with drug start: drug use = 1 and D_minus = 0
            new_row = pd.DataFrame([row['Carriage_ID'], drug_start - row['carriage_start'], 1]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Ineff_Drug_use']
            ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
            new_res_row = pd.DataFrame([row['Carriage_ID'], drug_start - row['carriage_start'], 0]).T
            new_res_row.columns = ['Carriage_ID', 'Time', 'D_minus']
            D_min_res_df = pd.concat([D_min_res_df, new_res_row], ignore_index=True)

            # add row if drug use ends before clearance
            if drug_end < row['carriage_end']:
                new_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 0]).T
                new_row.columns = ['Carriage_ID', 'Time', 'Ineff_Drug_use']
                ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
                new_res_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 1]).T
                new_res_row.columns = ['Carriage_ID', 'Time', 'D_minus']
                D_min_res_df = pd.concat([D_min_res_df, new_res_row], ignore_index=True)

        # 2. When drug effect is lost (= drug end) during colonisation event
        # drug_end is during colonisation but drug_start is before carriage starts - this is to avoid repeats
        drug_end_rows = data_df[(data_df['Infant'] == patient) &
                                (data_df['carriage_start'] <= drug_end) &
                                (data_df['carriage_end'] >= drug_end) &
                                (data_df['carriage_start'] >= drug_start)]

        for p, row in drug_end_rows.iterrows():
            # check whether drug is ineffective (has corresponding resistance)
            resistances_carried = [x for x in row['resistance_profile'].index.to_list()
                                   if row['resistance_profile'][x] == 1]
            if len(np.intersect1d(drug_resist, resistances_carried)) == 0:
                continue

            # add row with carriage start drug use = 1
            new_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 1]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Ineff_Drug_use']
            ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
            new_res_row = pd.DataFrame([row['Carriage_ID'], datetime.timedelta(0), 0]).T
            new_res_row.columns = ['Carriage_ID', 'Time', 'D_minus']
            D_min_res_df = pd.concat([D_min_res_df, new_res_row], ignore_index=True)

            # add row with drug end value changes to 0
            new_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 0]).T
            new_row.columns = ['Carriage_ID', 'Time', 'Ineff_Drug_use']
            ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
            new_res_row = pd.DataFrame([row['Carriage_ID'], drug_end - row['carriage_start'], 1]).T
            new_res_row.columns = ['Carriage_ID', 'Time', 'D_minus']
            D_min_res_df = pd.concat([D_min_res_df, new_res_row], ignore_index=True)

    ineffective_drug_df.drop(index=ineffective_drug_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    D_min_res_df.drop(index=D_min_res_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row

    # 3. For all other carriage_IDs not covered, the value of Drug_use co-variate is set to 0

    remaining_ids = list(set(data_df['Carriage_ID']) - set(ineffective_drug_df['Carriage_ID']))
    rem_df = pd.DataFrame({'Carriage_ID': remaining_ids, 'Time': datetime.timedelta(0), 'Ineff_Drug_use': 0})
    ineffective_drug_df = pd.concat([ineffective_drug_df, rem_df], ignore_index=True)
    ineffective_drug_df = ineffective_drug_df.assign(Time=pd.to_timedelta(ineffective_drug_df['Time']).dt.days.astype('int'), # Convert into integer format in days
                                                     Ineff_Drug_use=ineffective_drug_df.Ineff_Drug_use.astype('int'))
    
    # 3. For the remaining resistances, value of D_min = 1 when resistance and 0 otherwise
    subset = data_df[data_df['Carriage_ID'].isin(remaining_ids)]['resistance_profile']
    res_list = [s.sum() for s in subset]
    D_minus_list = [bool(x) if isinstance(x, float) else 0 for x in
                    res_list]  # Where resistance is not measured, still put zero
    rem_res = pd.DataFrame({'Carriage_ID': remaining_ids, 'Time': datetime.timedelta(0), 'D_minus': D_minus_list})
    D_min_res_df = pd.concat([D_min_res_df, rem_res], ignore_index=True)
    # Formatting:
    D_min_res_df = D_min_res_df.assign(Time=pd.to_timedelta(D_min_res_df['Time']).dt.days.astype('int'),
                                       D_minus=D_min_res_df.D_minus.astype('int')) # Convert into integer format in days

    # Dealing with duplicates and overlaps: Keep 0 when duplicates have 0 and 1
    ineffective_drug_df = ineffective_drug_df.loc[ineffective_drug_df.groupby(['Carriage_ID', 'Time'])['Ineff_Drug_use'].idxmin()]
    D_min_res_df = D_min_res_df.loc[D_min_res_df.groupby(['Carriage_ID', 'Time'])['D_minus'].idxmax()]
    
    return ineffective_drug_df, D_min_res_df


def add_average_durations(full_data, cocarriage_events):
    
    # Outputs an updated version of full_data with an additional column avgD
    # This is the average duration of the serotype carried in that episode of carriage, which
    # is calculated as the average of all observed durations that are singly-carried.
    
    ccIDs = cocarriage_events[cocarriage_events['Co_carried'] == 0][['Carriage_ID', 'Co_carried']]
    ccIDs = ccIDs.drop_duplicates() # Finds all singly-carried episodes
    
    full_data2 = full_data[full_data['Carriage_ID'].isin(ccIDs['Carriage_ID'])]
    # Keeps only those rows where there is no co-carriage
    # Using this we calculate average duration below
    avgD = full_data2.groupby('serotype_carried')['durations'].mean()
    
    # Now I need to add these calculated average durations to updated full_df
    updated_df = full_data.assign(avgD=full_data['serotype_carried'].map(avgD))
    
    # Similarly, I add the average durations of the co-carried serotypes
    max_avgD = pd.concat([cocarriage_events['OS1'].map(avgD),
                          cocarriage_events['OS2'].map(avgD),
                          cocarriage_events['OS3'].map(avgD)], axis=1).max(axis=1)

    updated_cc = cocarriage_events.assign(avgD_cc=max_avgD)
    
    return updated_df, updated_cc



# # COMMANDS TO GET ALL DATA:
# full_data, cocarriage_events, spec_dict = create_carriage_dataframe('Infant', 0.5)
# full_data, cocarriage_events = add_average_durations(full_data, cocarriage_events)
# full_data_with_moms, cc_with_moms = create_carriage_df_with_moms(full_data, cocarriage_events)
# illness_df = import_illness_data()
# simple_drug_df = get_simple_drug_df(full_data, illness_df, treat_length=7)
# effective_drug_df = effective_drug_plus_res(full_data, illness_df, treat_length=7)
# ineffective_drug_df, D_minus_df = ineffective_drugs(full_data, illness_df, treat_length=7)
# transmission_df = get_transmission_df(spec_dict, full_data, effective_drug_df, ineffective_drug_df)
