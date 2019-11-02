class Data:
    def __init__(self, condition=None, classification=-1):
        if condition is None:
            condition = []
        self.condition = condition
        self.classification = classification

class DataHelper:
    '''
    Class which helps parse the text file of data and sort it into relevant lists.
    '''
    data_list = []
    '''
    Extracts raw data from file into Data objects
    :param dataset_path - String: Absolute path to where a dataset should be.
    '''
    @staticmethod
    def load_file_data(dataset_path):
        with open(dataset_path, 'r') as f:
            for line in f:
                tempCondition = []
                for bit in line.split()[0]:
                    tempCondition.append(int(bit))
                DataHelper.data_list.append(
                    Data(
                        tempCondition, int(line.split()[1])))
