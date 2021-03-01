import numpy as np
import pandas as pd
from util import string2float


class FeatureExtractorCore:
    """
    Extract the features for forecasting
    """
    def __init__(self, df):
        self.df = df

    def get_dx(sef, row):

        dx_bl = row['DX_bl']
        if (dx_bl == 'CN'):
            dx_bl = 'NL'
        change_code = row['DXCHANGE']

        newdx = ''
        if (change_code in ['-1', 'NA']):
            newdx = dx_bl
        else:
            dd = {'1': 'NL', '2': 'MCI', '3': 'AD', '4': 'MCI', '5': 'AD', '6': 'AD', '7': 'NL', '8': 'MCI', '9': 'NL',
                  '-1': 'NA', 'NA': 'NA'}
            newdx = dd[change_code]

        row['DX_bl'] = newdx
        row['DXCHANGE'] = change_code
        return row


        # 1 = Stable:NL toNL,
        # 2 = Stable:MCI to MCI,
        # 3 = Stable:AD to AD,
        # 4 = Conv:NL to MCI,
        # 5 = Conv:MCI to AD,
        # 6 = Conv:NL to AD,
        # 7 = Rev:MCI to  NL,
        # 8 = Rev:AD to MCI,
        # 9 = Rev:AD to NL,\
        # -1 = Not available

    def conditional_replace_DX(self, row):
        if (row['DX'] != ''):
            if (row['DX'] in ['NL', 'CN']):  # NL
                # print 'hit'
                #return ['0']
                row['DX'] =0
            else:
                if row['DX'] == 'MCI':
                    #return ['1']
                    row['DX'] =1
                else:
                    #return ['2']
                    row['DX'] = 2
        return row


    def has_allcolumns(self,df, columns):
         temp =  [c in df.columns for c in columns]
         return  np.prod(temp).astype(np.bool)


    def extract_mri_features(self):
        """
        Extract relevant features from MRI feature sets
        :return:
        """
        cols = ['RID', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE',
                'Month_bl', 'ST101SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'ST102CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                'ST102SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 'AV45']
        cols.append('DX')
        cols.append('DX_bl')
        cols.append('DXCHANGE')

        #if (not self.has_allcolumns(self.df, cols)):
        #    raise ValueError('Some of the coluumns are not present')

        adf = self.df.loc[:, cols]
        print ('#Unique DX ', adf.DX.unique())
        print ('#Unique DX_bl ', adf.DX_bl.unique())
        print ('#Unique DXCHANGE ', adf.DXCHANGE.unique())

        adf[['DX', 'DXCHANGE']] = adf[['DX_bl', 'DXCHANGE']].apply(self.get_dx, axis=1)

        print ('#Diagnosis stats ', adf.groupby('DX').size())

        # set normal to 1 and abnormal to 0
        adf[['DX']] = adf[['DX']].apply(self.conditional_replace_DX, axis=1)

        # convert to numeric with NAN intact
        adf = adf.apply(pd.to_numeric, errors='coerce')


        # get rid of nan columns
        adf = adf[np.isfinite(adf['Ventricles'])]
        adf['Age_r'] = adf['Month_bl'] / 12.0 + adf['AGE']

        newcols = ['RID', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV',
         'Age_r',
         'DX', 'ST101SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
         'ST102CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
         'ST102SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']

        #if (not self.has_allcolumns(adf, newcols)):
        #    raise ValueError('Some of the coluumns are not present')


        feature_df = adf.loc[:,
                     newcols]

        feature_df = feature_df.dropna(how='any')

        print ('#Diagnosis stats  processed ', feature_df.groupby('DX').size())

        diagnosis = feature_df.pop('DX')
        group = feature_df.pop('RID')

        # whole_brain = feature_df.pop('WholeBrain')

        # feature_df = feature_df.div(whole_brain, axis=0)

        # feature_df['Age_r'] = adf['Month_bl'] / 12.0 + adf['AGE']


        X = feature_df.as_matrix().astype(np.float)
        y = diagnosis.as_matrix().astype(np.uint8)
        y = np.squeeze(y)

        return X, y, group


    def patientwise(self, derive_dx=True, include_col_list=None, genTest=False):
        """

        :param derive_dx:
        :param include_col_list: any additional features
        :param genTest: if True will generate  test data without generating label i.e for submission
        :return:
        """

        cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'Age_r',
                'DX']

        if (include_col_list is not None):
            cols.extend(include_col_list)
        genLabels = not genTest

        if(genLabels):
            if (derive_dx):
                self.df[['DX', 'DXCHANGE']] = self.df[['DX_bl', 'DXCHANGE']].apply(self.get_dx, axis=1)

            self.df[['DX']] = self.df[['DX']].apply(self.conditional_replace_DX, axis=1)

        self.df['Month_bl'] = pd.to_numeric(self.df['Month_bl'])
        self.df['AGE'] = pd.to_numeric(self.df['AGE'])
        self.df['Age_r'] = np.float32(self.df['Month_bl']) + 12 * np.float32(self.df['AGE'])

        eps = 0.0001
        self.df['ABETA_UPENNBIOMK9_04_19_17'] = np.log(
            pd.to_numeric(self.df['ABETA_UPENNBIOMK9_04_19_17'], errors='coerce') + eps)
        self.df['TAU_UPENNBIOMK9_04_19_17'] = np.log(
            pd.to_numeric(self.df['TAU_UPENNBIOMK9_04_19_17'], errors='coerce') + eps)
        self.df['PTAU_UPENNBIOMK9_04_19_17'] = np.log(
            pd.to_numeric(self.df['PTAU_UPENNBIOMK9_04_19_17'], errors='coerce') + eps)

        cols_missing = [not c in self.df.columns for c in cols]
        if(any(cols_missing)):
            miss_idx = np.where(np.asanyarray(cols_missing))[0]
            removed = [ cols.pop(id) for id in miss_idx]
            print('These columns are missing from dataframe and is removed', str(removed))





        features_mat = []
        labels_mat = []
        group_mat = []
        last_exam_date_mat = []
        groups_df = self.df.groupby('RID')
        groups = groups_df.groups  # this is a dictionary
        for pt in groups.keys():
            patienti = groups_df.get_group(pt)  # dataframe
            group_id = int(pt)

            # patienti['Month_bl'] = pd.to_numeric(patienti['Month_bl'])
            patienti = patienti.sort_values('Age_r')
            #Issue with .loc here
            examdates = patienti.loc[:, 'EXAMDATE'].values
            last_examdate = examdates[len(examdates) - 1]
            last_exam_date_mat.append(last_examdate)
            patient_mri = patienti.loc[:, cols]
            # patient_mri['Age_r'] = np.float32(patient_mri['Month_bl']) / 12.0 + np.float32(patient_mri['AGE'])

            if (genTest):
                features, isvalid = self.prepare_data_rf_mtp_test(patient_mri)
                labels = isvalid
            else:
                features, labels = self.prepare_data_rf_mtp(patient_mri)

            if (features.shape[0] > 0):
                group_ids = np.ones((features.shape[0],)) * group_id
                features_mat.append(features)
                labels_mat.append(labels)
                group_mat.append(group_ids)

                #print (features.shape, '-', labels.shape)

        return features_mat, labels_mat, group_mat, last_exam_date_mat


    def prepare_data_rf_mtp(self, pat_df):
        """

        :param pat:Sorted dataframe of a single patient
        :return:
        """

        def temp(arow0, arow1):
            # arow0= pat_df.iloc[i]
            # arow1= pat_df.iloc[i+1]

            features, suc = string2float(list(arow0))
            age1 = arow1['Age_r']
            features = np.append(features, age1)

            V0, isnum0 = string2float(arow0['Ventricles'])
            V1, isnum1 = string2float(arow1['Ventricles'])

            ICV0, isnumicv0 = string2float(arow0['ICV'])
            ICV1, isnumicv1 = string2float(arow1['ICV'])

            DX1 = int(arow1['DX'])

            labels = [V1 / (ICV1 + 0.000001), DX1]

            isvalid = isnum0 and isnum1 and isnumicv0 and isnumicv1

            return features, labels, isvalid

        features_mat = []
        labels_mat = []
        for i in range(0, len(pat_df) - 1):
            for j in range(i + 1, len(pat_df)):
                arowi = pat_df.iloc[i]
                arowj = pat_df.iloc[j]

                features, labels, isvalid = temp(arowi, arowj)
                if (isvalid):
                    features_mat.append(features)
                    labels_mat.append(labels)

        return np.asarray(features_mat), np.asarray(labels_mat)


    def prepare_data_rf_mtp_test(self, pat_df):
        """

        :param pat:Sorted dataframe of a single patient
        :return:
        """

        def temp(arow0):
            features, suc = string2float(list(arow0))
            # age1= arow1['Age_r']
            # features = np.append(features, age1)

            V0, isnum0 = string2float(arow0['Ventricles'])
            ICV0, isnumicv0 = string2float(arow0['ICV'])
            isvalid = isnum0 and isnumicv0
            return features, isvalid

        features_mat = []
        is_valid_mat = []

        for i in range(0, len(pat_df)):
            arowi = pat_df.iloc[i]
            features, isvalid = temp(arowi)
            features_mat.append(features)
            is_valid_mat.append(isvalid)

        return np.asarray(features_mat), np.asarray(is_valid_mat)


    def prepare_data_rf_stp(self, pat_df):
        """
         Single time point
        :param pat:Sorted dataframe of a single patient at single time point
        :return:
        """

        def temp(arow0):
            features, suc = string2float(list(arow0))

            V0, isnum0 = string2float(arow0['Ventricles'])
            isvalid = isnum0
            return features, isvalid

        features_mat = []
        labels_mat = []
        row0 = pat_df.iloc[0]
        features, isvalid = temp(row0)

        if (isvalid):
            features_mat.append(features)

        return np.asarray(features_mat), np.asarray(labels_mat)
