import const


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
    train_data = []
    test_data = []

    '''
    Extracts raw data from file into Data objects
    :param dataset_path - String: Absolute path to where a dataset should be.
    '''
    @staticmethod
    def load_file_data(dataset_path):
        temp_list = []
        with open(dataset_path, 'r') as f:
            for line in f:
                tempCondition = []
                lineSplit = line.split()
                i = 0
                for i in range(const.TRAIN_COND_LENGTH):
                    tempCondition.append(float(lineSplit[i]))
                temp_list.append(Data(
                        tempCondition, int(lineSplit[i+1])))
        DataHelper.train_data = temp_list[0:(len(temp_list) / 2)]
        DataHelper.test_data = temp_list[len(temp_list)/2:]
