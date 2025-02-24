# HERE, I DO THE SAME ANALYSIS AS IN ESTABLISHMENT_SURVIVAL.PY
# I ALTER THE DEFINITION OF TIME TO ESTABLISHMENT. NOW THIS MEANS: TIME FROM PREVIOUS EST. EVENT BY THE SAME SEROTYPE

import numpy as np
import pandas as pd
import datetime
from lifelines.utils import to_long_format
from lifelines.utils import add_covariate_to_timeline
from lifelines import CoxTimeVaryingFitter
from sklearn.preprocessing import MultiLabelBinarizer


def get_establishment_df(full_data, cocarriage_events):
    # When you look at establishment durations, we should do the regression on host properties, and NOT
    # on the properties of the strain that establishes. This is because regression on strain properties might just
    # indicate the prevalence of that property in the dataset ( Discussion on 17.05.2023 )

    # For each time-to-establish episode, I need the following properties:
    # (1) Host treatment status (2) presence of co-coloniser, (3) resistance of the co-coloniser (4) Age

    data_df = full_data.copy()
    co_data = cocarriage_events.copy()

    establishment_df = pd.DataFrame(
        columns=['Infant', 'Establish_ID', 'serotype', 'estab_start', 'estab_end', 'agestrat', 'durations',
                 'OS', 'observed',
                 'gen_immunity', 'sero_history'], index=[0])

    for index, carriage in data_df.iterrows():

        # Start of establishment is the previous carriage end time of SAME SEROTYPE. If there is no previous carriage,
        # then this is sampling start time and observed value is 0.
        current_carriage_start = data_df['carriage_start'][index]
        all_prev_carriages = data_df[(data_df['Infant'] == data_df['Infant'][index]) &
                                     (data_df['serotype_carried'] == data_df['serotype_carried'][index]) & # Same serotype
                                     (data_df['carriage_end'] < current_carriage_start)]
        if all_prev_carriages.empty:
            obs_value = 0   # This is for right censoring correction.
            estab_duration_start = data_df['samp_start'][index]
        else:
            obs_value = 1
            prev_ending = all_prev_carriages.loc[all_prev_carriages['carriage_end'].idxmax()]
            estab_duration_start = prev_ending['carriage_end']

        # I model co-carriage as time-varying covariate
        # and make a df for that in a separate function below. Here, I just
        # convert look at co-data to check whether there is a co-carried serotype at time
        # of establishment and what is its serotype. This is to ask the question:
        # does serotype of resident strain affect establishment rate of new strains.
        this_row = co_data[(co_data['Carriage_ID'] == data_df['Carriage_ID'][index]) &
                           (co_data['Time'] == 0)].reset_index()
        OS_list = [this_row['OS1'][0], this_row['OS2'][0], this_row['OS3'][0]]
        OS_list = [i for i in OS_list if i != 0]

        new_row = pd.DataFrame({'Infant': data_df['Infant'][index],
                                'Establish_ID': data_df['Carriage_ID'][index],
                                'serotype': data_df['serotype_carried'][index],
                                'estab_start': estab_duration_start,
                                'estab_end': current_carriage_start,
                                'age': data_df['age'][index],
                                'agestrat': data_df['agestrat'][index],
                                'durations': data_df['durations'][index],
                                'OS': [OS_list],
                                'observed': obs_value,
                                'gen_immunity1': data_df['gen_immunity1'][index],
                                'gen_immunity2': data_df['gen_immunity2'][index],
                                'sero_history1': data_df['sero_history1'][index],
                                'sero_history2': data_df['sero_history2'][index]
                                }, index=[index])

        establishment_df = pd.concat([establishment_df, new_row], axis=0)

    establishment_df.drop(index=establishment_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    establishment_df = establishment_df.replace(to_replace='nan',
                                                value=np.nan)  # For some reason, add_covariate_to_timeline()
    # function does not work properly with np.nan
    establishment_df = establishment_df.assign(
        e_duration=(establishment_df['estab_end'] - establishment_df['estab_start']).dt.days.astype('int'))

    return establishment_df


def find_co_res(df):
    if len(df) == 0:
        return pd.Series({'ceftriaxone': 0, 'chloramphenicol': 0,
                          'clindamycin': 0, 'co.trimoxazole': 0, 'erythromycin': 0,
                          'penicillin': 0, 'tetracycline': 0},
                         dtype='O')  # A zero-DF. When no co-carriage, this doesn't matter
    else:
        # If there are multiple carriages, check resistance in all of them.
        t = df['resistance_profile'].reset_index(drop=True)
        concatenated = pd.concat(t.tolist(), axis=1)
        concatenated = concatenated.dropna(axis=1)
        summed_series = concatenated.sum(axis=1)
        if not summed_series.empty:
            return summed_series.gt(0).astype(float) # Converts to binary
        else:     # In case the resistances are not measured in the data, send out an empty dataframe
            return pd.DataFrame(columns=['ceftriaxone', 'chloramphenicol', 'clindamycin', 'co.trimoxazole', 'erythromycin',
                                         'penicillin', 'tetracycline'])  # empty dataframe with columns


def get_establishment_co_carriage_df(establishment_df, full_data):
    # Here I make a dataframe for cocarriage in a format required by lifelines package.
    # I model co-carriage status as time-varying 0 or 1, since co-carriage at any time can contribute to
    # modifying the establishment rate.
    # Since time to establishment is defined as time from last serotype's clearance, there can be multiple
    # changes from 0 to 1 and 1 to 0 of co-carriage before establishment. 
    
    estab_df = establishment_df[['Infant', 'Establish_ID', 'estab_start', 'estab_end']].reset_index(drop=True)
    data_df = full_data.copy()
    
    # Predefine dataframe
    estab_host_carriages_df = pd.DataFrame(columns=['Establish_ID', 'Time', 'Host_carry', 'Co_res'], index=[0])

    for index, estab_event in estab_df.iterrows():
        # Relevant carriages are those who have a sero-gain/loss event happening during this establishment time.
        # Due to the way establishment event is defined, there can be multiple gain/loss during an estab_event.
        
        # Whenever a new carriage starts in the host during the time to establishment
        relevant_carriage_starts = data_df[(data_df['Infant'] == estab_event['Infant']) &
                                           (data_df['carriage_start'] < estab_event['estab_end']) &
                                           (data_df['carriage_start'] > estab_event['estab_start'])]
        
        # Whenever a carriage ends in the host during the time to establishment
        relevant_carriage_ends = data_df[(data_df['Infant'] == estab_event['Infant']) &
                                         (data_df['carriage_end'] < estab_event['estab_end']) &
                                         (data_df['carriage_end'] > estab_event['estab_start'])]
        
        # There may be a whole overarching host serotype without any events. Detect this first.
        # Check the carriage status at the start, see if there is another strain at start of
        # time to establishment. (It need not be overarching the whole time to establishment)
        occupied_at_start = data_df[(data_df['Infant'] == estab_event['Infant']) &
                                    (data_df['carriage_start'] <= estab_event['estab_start']) &
                                    (data_df['carriage_end'] > estab_event['estab_start'])]
        # Start of estab_event value
        new_row = pd.DataFrame([estab_event['Establish_ID'],
                                estab_event['estab_start'] - estab_event['estab_start'],  # Co-carry is at time 0
                                int(not occupied_at_start.empty), 0 #find_co_res(occupied_at_start)
                                ]).T
        new_row.columns = ['Establish_ID', 'Time', 'Host_carry', 'Co_res']
        estab_host_carriages_df = pd.concat([estab_host_carriages_df, new_row], ignore_index=True)
        
        # Then add the additional row for each gain/loss serotype events
        for index2, relevant_row in relevant_carriage_starts.iterrows():
            # At every new establishment, host-carriage status is set as 1
            date = relevant_row['carriage_start']+pd.Timedelta(days=1) # Right after the competor carriage began
            carriages_here = data_df[(data_df.carriage_start<date) &   
                                     (data_df.carriage_end>date)]    # All carriages in the host now
            new_row1 = pd.DataFrame([estab_event['Establish_ID'],
                                     relevant_row['carriage_start'] - estab_event['estab_start'],
                                     1, 0 #find_co_res(carriages_here)
                                     ]).T
            new_row1.columns = ['Establish_ID', 'Time', 'Host_carry', 'Co_res']
            estab_host_carriages_df = pd.concat([estab_host_carriages_df, new_row1], ignore_index=True)

        for index2, relevant_row in relevant_carriage_ends.iterrows():
            # At every clearance, host-carriage status MAY go back to 0
            date = relevant_row['carriage_end']+pd.Timedelta('1 day') # Right after clearance
            carriages_here = data_df[(data_df.carriage_start<date) &   
                                     (data_df.carriage_end>date)]    # Carriages still present after the clearance
            if carriages_here.empty:
                new_row2 = pd.DataFrame([estab_event['Establish_ID'],
                                        relevant_row['carriage_end'] - estab_event['estab_start'],
                                        int(not carriages_here.empty), 0 # find_co_res(carriages_here)
                                        ]).T
                new_row2.columns = ['Establish_ID', 'Time', 'Host_carry', 'Co_res']
                estab_host_carriages_df = pd.concat([estab_host_carriages_df, new_row2], ignore_index=True)
    
    # Final tidying up the dataframe to return      
    estab_host_carriages_df.drop(index=estab_host_carriages_df.index[0], axis=0,
                                 inplace=True)  # Getting rid of first Nan row
    estab_host_carriages_df = estab_host_carriages_df.assign(Time=pd.to_timedelta(estab_host_carriages_df['Time']).dt.days.astype('int'))
    estab_host_carriages_df = estab_host_carriages_df.reset_index(drop=True)

    return estab_host_carriages_df


def get_establishment_drug_use(estab_df, illness_df, treat_length=7):
    # I use drug consumption as a time-varying co-variate along an establishment episode length. This is
    # because any duration of drug effect can possibly elongate time till establishment.

    # ALGORITHM FOR DRUG:
    # I go through each illness episode (each row of illness_df) and check whether that illness episode (i.e. drug usage)
    # coincides with any establishment episode. There are three cases: (A) The illness episode is completely within
    # an episode, (B) The illness episodes starts in the middle of an establishment episode, (C) The illness
    # episode ends in the middle of the episode. For each case, the fraction of the establishment episode where
    # there is illness is marked with 1 for the drug column, and otherwise 0.

    estab_drug_use_df = pd.DataFrame(columns=['Establish_ID', 'Time', 'Drug_use'],
                                     index=[0])
    unique_drugs = pd.concat([illness_df['drug1'], illness_df['drug2'].dropna()]).unique()  # All drugs prescribed

    for index, drug_instance in illness_df.iterrows():

        patient = drug_instance['codenum']
        current_drugs = [x for x in unique_drugs if x in drug_instance[2:6].to_list()]  # Drugs under consumption

        drug_start = drug_instance['date_start']
        drug_end = drug_start + dt.timedelta(days=treat_length)  # Drug effect in host lasts 7 days

        # 1. When drug is taken during an establishment episode
        # drug_start date is sandwiched within the establishment start and end dates
        drug_start_rows = estab_df[(estab_df['Infant'] == patient) &
                                   (estab_df['estab_start'] <= drug_start) &
                                   (estab_df['estab_end'] >= drug_start)]  # relevant rows in this category

        for p, row in drug_start_rows.iterrows():
            # Add row with carriage start drug use = 0
            new_row = pd.DataFrame([row['Establish_ID'], dt.timedelta(0), 0]).T
            new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
            estab_drug_use_df = pd.concat([estab_drug_use_df, new_row], ignore_index=True)

            # add row with drug start
            new_row = pd.DataFrame([row['Establish_ID'], drug_start - row['estab_start'], 1]).T
            new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
            estab_drug_use_df = pd.concat([estab_drug_use_df, new_row], ignore_index=True)

            # add row with drug end (if ending before cleared)
            if drug_end < row['estab_end']:
                new_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 0]).T
                new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
                estab_drug_use_df = pd.concat([estab_drug_use_df, new_row], ignore_index=True)

        # 2. When drug effect is lost (= drug end) during establishment episode
        # drug_end is during estab episode but drug_start is before estab episode starts - this is to avoid repeats
        drug_end_rows = estab_df[(estab_df['Infant'] == patient) &
                                 (estab_df['estab_start'] <= drug_end) &
                                 (estab_df['estab_end'] >= drug_end) &
                                 (estab_df['estab_start'] >= drug_start)]

        for p, row in drug_end_rows.iterrows():
            # Add row with estab start drug use = 1
            new_row = pd.DataFrame([row['Establish_ID'], dt.timedelta(0), 1]).T
            new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
            estab_drug_use_df = pd.concat([estab_drug_use_df, new_row], ignore_index=True)

            # add row with drug end value changes to 0
            new_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 0]).T
            new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
            estab_drug_use_df = pd.concat([estab_drug_use_df, new_row], ignore_index=True)

        estab_drug_use_df.drop(index=estab_drug_use_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row

        # 3. For all other establish_IDs not covered, the value of Drug_use co-variate is set to 0
        remaining_ids = list(set(estab_df['Establish_ID']) - set(estab_drug_use_df['Establish_ID']))
        rem_df = pd.DataFrame({'Establish_ID': remaining_ids, 'Time': dt.timedelta(0), 'Drug_use': 0})
        estab_drug_use_df = pd.concat([estab_drug_use_df, rem_df], ignore_index=True)

    estab_drug_use_df = estab_drug_use_df.assign(Time=pd.to_timedelta(estab_drug_use_df['Time']).dt.days.astype('int'))  # Time when drug intake starts, in days
    
    return estab_drug_use_df


def get_est_drug_res_info(estab_df, full_data, illness_df, treat_length=7):
    
    # Prep the resistance data of the establishing strain
    estab_df = estab_df.merge(full_data[['Carriage_ID','resistance_profile']], left_on='Establish_ID', right_on='Carriage_ID', how='left')
    
    effective_drug_df = pd.DataFrame(columns=['Establish_ID', 'Time', 'Drug_use'],
                                     index=[0])
    ineffective_drug_df = pd.DataFrame(columns=['Establish_ID', 'Time', 'Ineff_Drug_use'],
                                     index=[0])
    res_without_drug = pd.DataFrame(columns=['Establish_ID', 'Time', 'D_minus'],
                                     index=[0])
    
    unique_drugs = pd.concat([illness_df['drug1'], illness_df['drug2'].dropna()]).unique()  # All drugs prescribed
    
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
    
    for index, drug_instance in illness_df.iterrows():   # Going through every illness episode

        patient = drug_instance['codenum']
        current_drugs = [x for x in unique_drugs if x in drug_instance[2:6].to_list()]  # Drugs under consumption
        drug_resist = [drug_dict[k] for k in current_drugs]  # Drugs converted to resistance format
        drug_start = drug_instance['date_start']
        drug_end = drug_start + datetime.timedelta(days=treat_length)  # Drug effect in host lasts 7 days

        # 1. When drug is taken during an establishment duration
        # drug_start date is sandwiched within the establishment start and end dates
        drug_start_rows = estab_df[(estab_df['Infant'] == patient) &
                                   (estab_df['estab_start'] <= drug_start) &
                                   (estab_df['estab_end'] >= drug_start)]  # relevant establishment rows in this category
        for p, row in drug_start_rows.iterrows():

            # check whether drug is effective or ineffective
            resistances_carried = [x for x in row['resistance_profile'].index.to_list()
                                   if row['resistance_profile'][x] == 1]
            
            if len(set(drug_resist) - set(resistances_carried)) == 0:
                # Ineffective drug use case    
                
                # Add row with carriage start: drug use = 0 and D_minus = 1
                new_row = pd.DataFrame([row['Establish_ID'], datetime.timedelta(0), 0]).T
                new_row.columns = ['Establish_ID', 'Time', 'Ineff_Drug_use']
                ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
                new_res_row = pd.DataFrame([row['Establish_ID'], datetime.timedelta(0), 1]).T
                new_res_row.columns = ['Establish_ID', 'Time', 'D_minus']
                res_without_drug = pd.concat([res_without_drug, new_res_row], ignore_index=True)

                # add row with drug start: drug use = 1 and D_minus = 0
                new_row = pd.DataFrame([row['Establish_ID'], drug_start - row['estab_start'], 1]).T
                new_row.columns = ['Establish_ID', 'Time', 'Ineff_Drug_use']
                ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
                new_res_row = pd.DataFrame([row['Establish_ID'], drug_start - row['estab_start'], 0]).T
                new_res_row.columns = ['Establish_ID', 'Time', 'D_minus']
                res_without_drug = pd.concat([res_without_drug, new_res_row], ignore_index=True)

                # add row if drug use ends before clearance
                if drug_end < row['estab_end']:
                    new_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 0]).T
                    new_row.columns = ['Establish_ID', 'Time', 'Ineff_Drug_use']
                    ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
                    new_res_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 1]).T
                    new_res_row.columns = ['Establish_ID', 'Time', 'D_minus']
                    res_without_drug = pd.concat([res_without_drug, new_res_row], ignore_index=True) 
                
            elif len(np.intersect1d(drug_resist, resistances_carried)) == 0:
                # effective drug use case
                # Add row with carriage start drug use = 0
                new_row = pd.DataFrame([row['Establish_ID'], datetime.timedelta(0), 0]).T
                new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
                effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

                # add row with drug start
                new_row = pd.DataFrame([row['Establish_ID'], drug_start - row['estab_start'], 1]).T
                new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
                effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

                # add row with drug end (if ending before cleared)
                if drug_end < row['estab_end']:
                    new_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 0]).T
                    new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
                    effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)
                    
        # 2. When drug effect is lost (= drug end) during establishment duration event
        # drug_end is during estab-D but drug_start is before estab starts - this is to avoid repeats
        drug_end_rows = estab_df[(estab_df['Infant'] == patient) &
                                 (estab_df['estab_start'] <= drug_end) &
                                 (estab_df['estab_end'] >= drug_end) & # Maybe i don't need this line
                                 (estab_df['estab_start'] >= drug_start)]
        for p, row in drug_end_rows.iterrows():
            # check whether drug is effective (no resistance)
            resistances_carried = [x for x in row['resistance_profile'].index.to_list()
                                   if row['resistance_profile'][x] == 1]
            
            if len(set(drug_resist) - set(resistances_carried)) == 0:
                # Ineffective drug use case  
                # add row with carriage start drug use = 1
                new_row = pd.DataFrame([row['Establish_ID'], datetime.timedelta(0), 1]).T
                new_row.columns = ['Establish_ID', 'Time', 'Ineff_Drug_use']
                ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
                new_res_row = pd.DataFrame([row['Establish_ID'], datetime.timedelta(0), 0]).T
                new_res_row.columns = ['Establish_ID', 'Time', 'D_minus']
                res_without_drug = pd.concat([res_without_drug, new_res_row], ignore_index=True)

                # add row with drug end value changes to 0
                new_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 0]).T
                new_row.columns = ['Establish_ID', 'Time', 'Ineff_Drug_use']
                ineffective_drug_df = pd.concat([ineffective_drug_df, new_row], ignore_index=True)
                new_res_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 1]).T
                new_res_row.columns = ['Establish_ID', 'Time', 'D_minus']
                res_without_drug = pd.concat([res_without_drug, new_res_row], ignore_index=True)
                
            elif len(np.intersect1d(drug_resist, resistances_carried)) == 0:
                # Effective drug use case
                # add row with carriage start drug use = 1
                new_row = pd.DataFrame([row['Establish_ID'], datetime.timedelta(0), 1]).T
                new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
                effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)

                # add row with drug end value changes to 0
                new_row = pd.DataFrame([row['Establish_ID'], drug_end - row['estab_start'], 0]).T
                new_row.columns = ['Establish_ID', 'Time', 'Drug_use']
                effective_drug_df = pd.concat([effective_drug_df, new_row], ignore_index=True)
    
    effective_drug_df.drop(index=effective_drug_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    ineffective_drug_df.drop(index=ineffective_drug_df.index[0], axis=0, inplace=True)  # Getting rid of first Nan row
    res_without_drug.drop(index=res_without_drug.index[0], axis=0, inplace=True)
        
    # 3. For all other carriage_IDs not covered, both drug use is zero
    remaining_ids_i = list(set(estab_df['Establish_ID']) - set(ineffective_drug_df['Establish_ID']))
    rem_df = pd.DataFrame({'Establish_ID': remaining_ids_i, 'Time': datetime.timedelta(0), 'Ineff_Drug_use': 0})
    ineffective_drug_df = pd.concat([ineffective_drug_df, rem_df], ignore_index=True)
    ineffective_drug_df = ineffective_drug_df.assign(Time=pd.to_timedelta(ineffective_drug_df['Time']).dt.days.astype('int'), # Convert into integer format in days
                                                        Ineff_Drug_use=ineffective_drug_df.Ineff_Drug_use.astype('int'))
    
    remaining_ids_e = list(set(estab_df['Establish_ID']) - set(effective_drug_df['Establish_ID']))
    rem_df = pd.DataFrame({'Establish_ID': remaining_ids_e, 'Time': datetime.timedelta(0), 'Drug_use': 0})
    effective_drug_df = pd.concat([effective_drug_df, rem_df], ignore_index=True)
    effective_drug_df = effective_drug_df.assign(Time=pd.to_timedelta(effective_drug_df['Time']).dt.days.astype('int'),
                                                    Drug_use=effective_drug_df['Drug_use'].astype('int'))
    
    # 3. For the remaining resistances, value of D_min = 1 when resistance and 0 otherwise
    subset = estab_df[estab_df['Establish_ID'].isin(remaining_ids_i)]['resistance_profile']
    res_list = [s.sum() for s in subset]
    D_minus_list = [bool(x) if isinstance(x, float) else 0 for x in
                    res_list]  # Where resistance is not measured, still put zero
    rem_res = pd.DataFrame({'Establish_ID': remaining_ids_i, 'Time': datetime.timedelta(0), 'D_minus': D_minus_list})
    res_without_drug = pd.concat([res_without_drug, rem_res], ignore_index=True)
    # Formatting:
    res_without_drug = res_without_drug.assign(Time=pd.to_timedelta(res_without_drug['Time']).dt.days.astype('int'),
                                               D_minus=res_without_drug.D_minus.astype('int')) # Convert into integer format in days

    # Dealing with duplicates and overlaps:
    ineffective_drug_df = ineffective_drug_df.loc[ineffective_drug_df.groupby(['Establish_ID', 'Time'])['Ineff_Drug_use'].idxmin()]
    res_without_drug = res_without_drug.loc[res_without_drug.groupby(['Establish_ID', 'Time'])['D_minus'].idxmax()]
    effective_drug_df = effective_drug_df.loc[effective_drug_df.groupby(['Establish_ID', 'Time'])['Drug_use'].idxmax()]

    return effective_drug_df, ineffective_drug_df, res_without_drug
        
        
# ____________________________________________________________________________________________________________________
# SURVIVAL ANALYSIS:
# ____________________________________________________________________________________________________________________


def estab_survival_cox_basic(establishment_df, estab_host_carriages_df):
    # QUESTION:
    # Does the presence of a co-colonizer in the host affect establishment of a new strain?

    # EQUATION:
    # TIME TO ESTABLISHMENT ~ CO-CARRIAGE STATE

    data_df = establishment_df[['Establish_ID', 'agestrat', 'observed', 'e_duration', 'serotype']].reset_index(drop=True)
    co_carry_df = estab_host_carriages_df[['Establish_ID', 'Time', 'Host_carry']].reset_index(drop=True)

    data_df = to_long_format(data_df, duration_col="e_duration")  
    req_df = add_covariate_to_timeline(data_df, co_carry_df, duration_col='Time', id_col='Establish_ID',
                          event_col='observed')
    req_df = req_df.dropna()
    
    estab_cox = CoxTimeVaryingFitter(penalizer=0.01)
    estab_cox.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                  strata=['agestrat', 'serotype'], show_progress=True)

    return estab_cox


def estab_survival_sero_competition(establishment_df, estab_host_carriages_df):
    # QUESTION:
    # Does the serotype of resident strain in the host affect establishment of a new strain?

    # EQUATION:
    # TIME TO ESTABLISHMENT ~ CO-CARRIAGE STATE + RESIDENT-SEROTYPE

    data_df = establishment_df[['Establish_ID', 'agestrat', 'observed', 'serotype', 'e_duration', 'OS']]
    co_carry_df = estab_host_carriages_df[['Establish_ID', 'Time', 'Host_carry']]

    # What I do in below block: create dummies for co-carried serotypes
    mlb = MultiLabelBinarizer()
    array_out = mlb.fit_transform(data_df['OS'])
    dummies = pd.DataFrame(data=array_out, columns=mlb.classes_)
    dummies.index = data_df.index
    data_df = data_df.join(dummies)
    data_df = data_df.drop(['OS', '18C'], axis=1)
    removed_sero = dummies.columns[(dummies.sum() <= 4)]  # Low frequency serotypes to remove from regression

    data_df = to_long_format(data_df, duration_col="e_duration")
    req_df = add_covariate_to_timeline(data_df, co_carry_df, duration_col="Time", id_col="Establish_ID",
                                       event_col="observed")
    req_df = req_df.dropna()

    req_df = req_df.loc[(req_df[removed_sero] != 1).all(axis=1)]
    # Remove those rows which are serotypes in removed_sero as these were removed when low frequency dummy variables were removed
    req_df = req_df.drop(removed_sero, axis=1)  # Remove dummy variables (columns in removed_sero)

    estab_cox = CoxTimeVaryingFitter(penalizer=0.01)
    estab_cox.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                  strata=['agestrat', 'serotype'], show_progress=False)

    return estab_cox


def estab_survival_drugs(establishment_df, estab_host_carriages_df, estab_drug_use_df):
    # QUESTION:
    # Does drug consumption by the host affect establishment of a new strain?

    # EQUATION:
    # TIME TO ESTABLISHMENT ~  CO-CARRIAGE STATE + HOST-DRUG-USE

    data_df = establishment_df[['Establish_ID', 'agestrat', 'observed', 'serotype', 'e_duration']]
    drug_df = estab_drug_use_df.copy()
    co_carry_df = estab_host_carriages_df[['Establish_ID', 'Time', 'Host_carry']]
    
    data_df = to_long_format(data_df, duration_col="e_duration")
    req_df = data_df.pipe(add_covariate_to_timeline,co_carry_df, duration_col="Time", id_col="Establish_ID",
                          event_col="observed") \
        .pipe(add_covariate_to_timeline, drug_df, duration_col="Time", id_col="Establish_ID", event_col="observed")
    req_df = req_df.dropna()

    estab_cox = CoxTimeVaryingFitter(penalizer=0.01)
    estab_cox.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                  strata=['agestrat', 'serotype'], show_progress=True)

    return estab_cox


def estab_survival_res_and_drugs(establishment_df, estab_host_carriages_df, e_effective_drug_df,
                                 e_ineffective_drug_df):
    # QUESTION:
    # Does effective-drug-consumption by the host affect establishment of a new strain?

    # EQUATION:
    # TIME TO ESTABLISHMENT ~ CO-CARRIAGE STATE + INEFF_DRUG_USE + EFFECTIVE_DRUG_USE

    data_df = establishment_df[['Establish_ID', 'agestrat', 'observed', 'serotype', 'e_duration']]
    co_carry_df = estab_host_carriages_df[['Establish_ID', 'Time', 'Host_carry']]
    drug_eff = e_effective_drug_df.copy()
    drug_ineff = e_ineffective_drug_df.copy()
    # data_df = data_df.assign(
    #     res=data_df['resistance_profile'].apply(lambda x: x['tetracycline'] if x.empty == 0 else np.nan))
    
    data_df = to_long_format(data_df, duration_col="e_duration")
    req_df = data_df.pipe(add_covariate_to_timeline,drug_eff, duration_col="Time", id_col="Establish_ID", event_col="observed") \
                    .pipe(add_covariate_to_timeline, drug_ineff, duration_col="Time", id_col="Establish_ID", event_col="observed") \
                    .pipe(add_covariate_to_timeline, co_carry_df, duration_col="Time", id_col="Establish_ID", event_col="observed")
    req_df = req_df.dropna()

    # req_df = req_df.drop(['resistance_profile', 'res'], axis=1)
    estab_cox = CoxTimeVaryingFitter(penalizer=0.01)
    estab_cox.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                  strata=['agestrat', 'serotype'], show_progress=True)

    return estab_cox


def estab_survival_resident_resist(establishment_df, estab_host_carriages_df, drug):
    # QUESTION:
    # Does resistant competitor within host affect establishment rate of incoming strains?

    # EQUATION:
    # TIME TO ESTABLISHMENT ~ RESISTANT-COMPETITOR + SENSITIVE-COMPETITOR

    data_df = establishment_df[['Establish_ID', 'agestrat', 'observed', 'serotype', 'e_duration']]
    co_carry_df = estab_host_carriages_df.copy()

    if drug == 'Any':
        co_carry_df = co_carry_df.assign(
            Res=co_carry_df['Co_res'].reset_index(drop=True).apply(lambda x: x.max() if x.empty == 0 else np.nan))
    else:
        co_carry_df = co_carry_df.assign(
            Res=co_carry_df['Co_res'].reset_index(drop=True).apply(lambda x: x[drug] if x.empty == 0 else np.nan))

    co_carry_df = co_carry_df.drop('Co_res', axis=1)
    data_df = to_long_format(data_df, duration_col="e_duration")
    req_df = data_df.pipe(add_covariate_to_timeline, co_carry_df, duration_col="Time", id_col="Establish_ID",
                          event_col="observed")
    req_df = req_df.dropna()
    
    req_df = req_df.assign(CC_Res=lambda x: (x['Host_carry'] == 1) & (x['Res'] == 1),
                           CC_Sen=lambda x: (x['Host_carry'] == 1) & (x['Res'] == 0))
    req_df = req_df.assign(CC_Res=req_df.CC_Res.astype('int'),
                           CC_Sen=req_df.CC_Sen.astype('int'))
    req_df = req_df.drop(['Res', 'Host_carry'], axis=1)
    req_df = req_df.dropna()
    
    est_res = CoxTimeVaryingFitter(penalizer=0.01)
    est_res.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                formula='CC_Res+CC_Sen',
                strata=['agestrat', 'serotype'], show_progress=False)

    return est_res


def estab_withinhostcost_resistance(establishment_df, estab_host_carriages_df,
                                    e_effective_drug_df, e_ineffective_drug_df,
                                    res_without_drug_df):
    # Is there a competitive cost of resistance during establishment? In other words:
    # If the incoming strain is resistant, does it have a longer time to establishment
    # than an incoming sensitive strain?
    # Here, we look at the interaction term RES X presence-of-competitor
    
    # EQUATION:
    # TIME TO ESTABLISHMENT ~ CO-CARRIAGE STATE + INEFF_DRUG_USE + EFFECTIVE_DRUG_USE
    #                           + RES_WITHOUT_DRUG + RES X CO-CARRIAGE STATE
    # HERE RES & DRUGS WITH RESPECT TO THE INCOMING (FOCAL) STRAIN
    
    data_df = establishment_df[['Establish_ID', 'agestrat', 'observed', 'serotype', 'e_duration']]
    co_carry_df = estab_host_carriages_df[['Establish_ID', 'Time', 'Host_carry']]
    drug_eff = e_effective_drug_df.copy()
    drug_ineff = e_ineffective_drug_df.copy()
    res_df = res_without_drug_df.copy()
    
    data_df = to_long_format(data_df, duration_col="e_duration")
    req_df = data_df.pipe(add_covariate_to_timeline,drug_ineff, duration_col="Time", id_col="Establish_ID", event_col="observed") \
                    .pipe(add_covariate_to_timeline, drug_eff, duration_col="Time", id_col="Establish_ID", event_col="observed")\
                    .pipe(add_covariate_to_timeline, res_df, duration_col="Time", id_col="Establish_ID", event_col="observed")\
                    .pipe(add_covariate_to_timeline, co_carry_df, duration_col="Time", id_col="Establish_ID", event_col="observed")
    req_df = req_df.dropna()
    
    equation = 'Host_carry+Drug_use+Ineff_Drug_use+D_minus+D_minus:Host_carry'
    estab_cox = CoxTimeVaryingFitter(penalizer=0.01)
    estab_cox.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                  formula=equation,
                  strata=['agestrat', 'serotype'], show_progress=True)
    
    return estab_cox


def estab_survival_immunity(establishment_df, estab_host_carriages_df):
    # QUESTION:
    # Does immunity affect establishment rate of a new strain?

    # EQUATION:
    # TIME TO ESTABLISHMENT ~ CO-CARRIAGE STATE + GEN-IMMUNITY + SERO-IMMUNITY

    data_df = establishment_df[['Establish_ID', 'age', 'observed', 'e_duration', 'serotype',
                                'gen_immunity1', 'sero_history1', 'gen_immunity2', 'sero_history2']].reset_index(drop=True)
    co_carry_df = estab_host_carriages_df[['Establish_ID', 'Time', 'Host_carry']].reset_index(drop=True)

    data_df = to_long_format(data_df, duration_col="e_duration")
    req_df = add_covariate_to_timeline(data_df, co_carry_df, duration_col='Time', id_col='Establish_ID',
                                       event_col='observed')
    req_df = req_df.dropna()

    estab_cox = CoxTimeVaryingFitter(penalizer=0.01)
    estab_cox.fit(req_df, id_col="Establish_ID", event_col="observed", start_col="start", stop_col="stop",
                  strata=['serotype'], show_progress=True)

    return estab_cox
