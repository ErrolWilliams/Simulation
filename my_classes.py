import numpy as np
import keras


class DataGenerator_labels(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, nbatch=100, dim=(20, 64, 64), n_channels=5,
                 shuffle=True, load_path='data/upDown1234/'):
        'Initialization'
        self.dim = dim
        self.nbatch = nbatch
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_digits = n_channels - 1
        self.shuffle = shuffle
        self.load_path = load_path
        self.labels = labels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.nbatch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.nbatch:(index + 1) * self.nbatch]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __preprocess(self, X, digits):
        X = X.astype('float32')
        finV = np.ones((X.shape[0], X.shape[1], X.shape[2], self.n_digits))
        finO = np.zeros((X.shape[0], X.shape[1], X.shape[2], self.n_digits))
        # Copy visibility grid (originally position 1) and paste it on 0
        finV[:, :, :, (digits[0] - 1)] = np.copy(X[:, :, :, 1])
        finV[:, :, :, (digits[0] - 1)] = np.copy(X[:, :, :, 1])
        # copy occupancy grids
        mn1 = np.copy(X[:, :, :, 0])
        mn2 = np.copy(X[:, :, :, 0])
        mn1[mn1 == digits[1]] = 0
        mn2[mn2 == digits[0]] = 0
        finO[:, :, :, (digits[0] - 1)] = mn1
        finO[:, :, :, (digits[1] - 1)] = mn2
        finO[finO > 0] = finO[finO > 0] / finO[finO > 0]
        mask = np.reshape(np.copy(X[:, :, :, 1]), (X.shape[0], X.shape[1], X.shape[2], 1))
        return np.concatenate((finO, mask, finV), axis=3)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.nbatch, *self.dim, 2))
        # X = np.empty((self.nbatch, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        labs = self.labels[list_IDs_temp]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(self.load_path + str(ID) + '.npy')

        X = np.stack([self.__preprocess(X[i, :, :, :, :], labs[i]) for i in range(self.nbatch)], axis=0)
        # Y recreates all occupancies, (not visibility: -1)
        Y = np.copy(X[:, 10:, :, :, :self.n_digits]).reshape((self.nbatch, 10, self.dim[1], self.dim[2], self.n_digits))
        return X, Y


class DataGenerator_flat(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, nbatch=100, dim=(2, 20, 64, 64), n_channels=2,
                 shuffle=True, load_path='data/upDown1234/'):
        'Initialization'
        self.dim = dim
        self.nbatch = nbatch
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_path = load_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.nbatch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.nbatch:(index + 1) * self.nbatch]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __preprocess(self, X):
        X = X.astype('float32')
        X[X > 0] = X[X > 0] / X[X > 0]
        # Y is the occupancy grid on X (position 0), last 10 frames
        Y = np.copy(X[:, 10:, :, :, 0]).reshape((self.nbatch, 10, self.dim[1], self.dim[2], 1))
        return X, Y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        X = np.empty((self.nbatch, *self.dim, 2))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load(self.load_path + str(ID) + '.npy')
        X, Y = self.__preprocess(X)
        return X, Y


class DataGenerator_flat_multi(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, nbatch=100, dim=(20, 50, 50), n_channels=2,
                 shuffle=True, load_path=['data/upDown1234/']):
        'Initialization'
        self.dim = dim
        self.nbatch = nbatch
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.load_path = load_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.nbatch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.nbatch:(index + 1) * self.nbatch]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X, Y = self.__data_generation(list_IDs_temp, )

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __preprocess(self, X):
        X = X.astype('float32')
        X[X > 0] = X[X > 0] / X[X > 0]
        # Y1 is the occupancy grid on X1 (position 0), last 10 frames
        Y = np.copy(X[:, 10:, :, :, 0]).reshape((self.nbatch, 10, self.dim[1], self.dim[2], 1))
        return X, Y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        X1 = np.empty((self.nbatch, *self.dim, 2))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            data = np.zeros((20, 50, 50, 2))
            data = data.astype(np.int, copy=False)
            for j in range(len(self.load_path)):
                temp_data = np.load(self.load_path[j] + str(ID) + '.npy')
                data = np.bitwise_or(data, temp_data)
            X1[i,] = data
        X, Y = self.__preprocess(X1)
        return X, Y


# My classes
# REMEMBER:
# modify data directory
class DataGenerator_col(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, nbatch=100, dim=(20, 64, 64), n_channels=2, dim_out=(10, 64, 64),
                 shuffle=True, load_path_u='data/final4/up/', load_path_d='data/final4/down/',
                 load_path_y='data/final4/Y/'):
        'Initialization'
        self.dim = dim
        self.dim_out = dim_out
        self.nbatch = nbatch
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.load_path_u = load_path_u
        self.load_path_d = load_path_d
        self.load_path_y = load_path_y
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.nbatch))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.nbatch:(index + 1) * self.nbatch]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # here X is a list of 2 arrays
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        Xu = np.empty((self.nbatch, *self.dim, 2))
        Xd = np.empty((self.nbatch, *self.dim, 2))
        Y = np.empty((self.nbatch, *self.dim_out, 1))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            Xu[i,] = np.load(self.load_path_u + str(ID) + '.npy')
            Xd[i,] = np.load(self.load_path_d + str(ID) + '.npy')
            Y[i,] = np.load(self.load_path_y + str(ID) + '.npy')
        Xu = Xu.astype(np.float32)
        Xd = Xd.astype(np.float32)
        Y = Y.astype(np.float32)
        return [Xu, Xd], Y
