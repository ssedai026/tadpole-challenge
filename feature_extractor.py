from util import read_csv_file
from util import load_csv_as_dataframe
import numpy as np
from feature_extractor_core import FeatureExtractorCore


class FeatureExtractor:
    """
    Extract features and save as numpy file for training  the models
    """

    def __init__(self, dict_file, data_file_csv=None, data=None):
        assert data_file_csv is not None or data is not None, 'at least one of datafile or data should be present'

        self.dict_file = dict_file

        self.col_indices = []

        self.dict = load_csv_as_dataframe(self.dict_file)

        if (data_file_csv is not None):
            self.data = load_csv_as_dataframe(data_file_csv)
        else:
            self.data = data

        self.fe = FeatureExtractorCore(self.data)

        self.add_extra_cols()

    def add_extra_cols(self):
        # self.add_column(range(485, 831))
        self.add_column([1138])  # ab45
        self.add_column(range(1139, 1163))
        self.add_column([24, 25, 26, 27, 28])  # cog scores
        self.add_column([1916, 1918])  # csf

    def prepare_data_train(self, derive_dx=True):

        #added 20jan
        #derive_dx = False #True #for d3 derive_dx=false

        extra_cols = self.col_names()
        features, labels, groups, lastexamdate = self.fe.patientwise(derive_dx, extra_cols, genTest=False)

        return np.vstack(features), np.vstack(labels), np.concatenate(groups)

    def prepare_data_test(self):
        # added 20jan
        derive_dx = False  # True #for d3 derive_dx=false

        extra_cols = self.col_names()
        features, isvalid, groups, lastexamdate = self.fe.patientwise(derive_dx, extra_cols, genTest=True)

        for f, v, g, ed in zip(features, isvalid, groups, lastexamdate):
            yield f, v, g[0], ed

    def add_column(self, indices):
        indices = np.asarray(indices)
        indices[0] = indices[0] - 1
        indices = indices - 1
        self.col_indices.extend(indices)  # -1 because indices are including header

    def col_names(self):
        fields = self.dict['FLDNAME']
        indices = np.asarray(self.col_indices)
        fields_name = list(fields[indices])
        exc_list = self.get_exclusion_list()

        final_list = set(fields_name) - set(exc_list)

        return sorted (list(final_list))

    def get_exclusion_list(self):

        el = ['ST122SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST100SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16',
              'ST126SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST22CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST22SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST22TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST22TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST33SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST41SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST63SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST67SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST81CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST81SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST81TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST81TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST87CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST8SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
              'ST92SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']
        return el


if (__name__ == '__main__'):
    dp = FeatureExtractor('data/TADPOLE_D1_D2_Dict.csv', 'data/d1_data.csv')

    cols = dp.col_names()
    print (len(cols))
    print (cols[0], '    ', cols[len(cols) - 1])

    features, labels, groups = dp.prepare_data_train()
    print (features.shape)
    np.savez('data/cur_data_d1.npz', features=features, labels=labels, groups=groups)

