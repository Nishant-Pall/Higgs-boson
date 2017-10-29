import numpy as np

def standardize(x, mean, std):
    '''
    Normalizes values in columns.
    '''
    if mean is not None:
        x = (x - mean) / std
        return x, mean, std
    else:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x-mean)/std
        return x, mean, std

def del_inavlid_values(set):
    """
    Deletes columns that have all invalid values. Invalid
    values are ones equal to '-999'.
    """
    del_col = []

    for i in range(len(set[0])):
        invalid = (set[:, i] == -999.000)
        if (invalid.all() == True):
            del_col.append(i)

    set = np.delete(set, del_col, axis=1)
    return set

def separate_invalid(set):
    '''
    Splits one data set into two, based on the value in column 'DER_mass_MMC'. 
    Split takes into account whether observed value is valid or invalid. 
    Invalid values are ones equal to '-999'.
    '''
    set_invalid = set[set[:, 2] < 0]
    set_none_invalid = set[set[:, 2] > 0]

    return set_none_invalid, np.delete(set_invalid, [2], axis=1)

def get_sets(data):
    '''
    Splits given set (train or test) into four new sets, based on the value in 
    column 'PRI_jet_num'.
    '''
    train_set_0 = data[data[:, 24] == 0]
    train_set_1 = data[data[:, 24] == 1]
    train_set_2 = data[data[:, 24] == 2]
    train_set_3 = data[data[:, 24] == 3]

    # For every set, delete column 24 which is 'PRI_jet_num'.
    # For train_set_0 where all 'PRI_jet_num' are equal to '0', delete also 
    # column 5('DER_pt_h') because 'DER_pt_h' is always equal to 'DER_pt_tot'.
    # Also for train_set_0 delete column 31 ('PRI_jet_all_pt') because all values are equal to '0'.
    train_set_0 = np.delete(train_set_0, [5, 24, 31], axis=1)

    # For train_set_1 delete column 31 ('PRI_jet_all_pt') because 
    # 'PRI_jet_all_pt' is always equal to 'PRI_jet_leading_pt'.
    train_set_1 = np.delete(train_set_1, [24, 31], axis=1)

    # For train_set_2 delete column 31 ('PRI_jet_all_pt') because 
    # 'PRI_jet_all_pt' is always the sum of 'PRI_jet_leading_pt' and 
    # 'PRI_jet_subleading_pt'.
    # For train_set_2 and train_set_3 delete column 6('DER_deltaeta_jet_jet') 
    # because 'DER_deltaeta_jet_jet' is always absolute value of the difference 
    # between 'PRI_jet_leading_eta' and 'PRI_jet_subleading_eta'.
    train_set_2 = np.delete(train_set_2, [6, 24, 31], axis=1)
    train_set_3 = np.delete(train_set_3, [6, 24], axis=1)

    train_set_0 = del_inavlid_values(train_set_0)
    train_set_1 = del_inavlid_values(train_set_1)
    train_set_2 = del_inavlid_values(train_set_2)
    train_set_3 = del_inavlid_values(train_set_3)

    train_set_0, train_set_0_1 = separate_invalid(train_set_0)
    train_set_1, train_set_1_1 = separate_invalid(train_set_1)
    train_set_2, train_set_2_1 = separate_invalid(train_set_2)
    train_set_3, train_set_3_1 = separate_invalid(train_set_3)
    
    return [train_set_0, train_set_0_1, train_set_1, train_set_1_1, train_set_2, train_set_2_1, train_set_3, train_set_3_1]