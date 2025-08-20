import tensorflow as tf
import numpy as np
import gc

from senkin_tmp_cite_pred.metrics import correlation_score, cosine_similarity_loss

def zscore(x):
    x_zscore = []
    for i in range(x.shape[0]):
        x_row = x[i]
        x_row = (x_row - np.mean(x_row)) / np.std(x_row)
        x_zscore.append(x_row)
    x_std = np.array(x_zscore)    
    return x_std

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_X, train_y, list_IDs, shuffle, batch_size, labels, ): 
        super().__init__()
        self.train_X = train_X
        self.train_y = train_y
        self.list_IDs = list_IDs        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = labels
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        ct = len(self.list_IDs) // self.batch_size
        return ct
    
    def __getitem__(self, idx):
        'Generate one batch of data'
        indexes = np.arange(idx*self.batch_size, (idx+1)*self.batch_size)

        X, y = self.train_X[indexes], self.train_y[indexes]

        if self.labels: return X, y
        else: return X
 
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange( len(self.list_IDs) )
        if self.shuffle: 
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'    
        X = self.train_X[list_IDs_temp]
        y = self.train_y[list_IDs_temp]        
        return X, y
    
def nn_kfold(train_cell_ids, train_cite_X, train_cite_y, test_cell_ids, test_cite_X, network, folds, model_name, BATCH_SIZE, EPOCHS, LR_FACTOR):
    train_preds = np.zeros((len(train_cell_ids), train_cite_y.shape[1]))
    test_preds = np.zeros((len(test_cell_ids), train_cite_y.shape[1]))
    cv_corr = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_cell_ids,)):          
        print (n_fold)
        train_x = train_cite_X[train_idx]
        valid_x = train_cite_X[valid_idx]
        train_y = train_cite_y[train_idx]
        valid_y = train_cite_y[valid_idx]

        train_x_index = train_cell_ids[train_idx]
        valid_x_index = train_cell_ids[valid_idx]

        model = network(train_cite_X.shape[1], n_targets=train_cite_y.shape[1])
        filepath = "models/" + model_name + '_' + str(n_fold) + '.weights.h5'
        es = tf.keras.callbacks.EarlyStopping(patience=10, mode='min', verbose=1) 
        checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True,save_weights_only=True,mode='min') 
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR, patience=6, verbose=1)
    
        train_dataset = DataGenerator(
            train_x,
            train_y,
            list_IDs=train_x_index, 
            shuffle=True, 
            batch_size=BATCH_SIZE, 
            labels=True,
        )
        
        valid_dataset = DataGenerator(
            valid_x,
            valid_y,
            list_IDs=valid_x_index, 
            shuffle=False, 
            batch_size=BATCH_SIZE, 
            labels=True,
        )
    
        hist = model.fit(train_dataset,
                        validation_data=valid_dataset,  
                        epochs=EPOCHS, 
                        callbacks=[checkpoint,es,reduce_lr_loss],
                        verbose=1)  
    
        model.load_weights(filepath)
        
        train_preds[valid_idx] = model.predict(valid_x, 
                                batch_size=BATCH_SIZE,
                                verbose=1)
        
        oof_corr = correlation_score(valid_y,  train_preds[valid_idx])
        cv_corr.append(oof_corr)
        print (cv_corr)       
        
        test_preds += model.predict(test_cite_X, 
                                batch_size=BATCH_SIZE,
                                verbose=1) / folds.n_splits 
            
        del model
        gc.collect()
        tf.keras.backend.clear_session()
    cv = correlation_score(train_cite_y, train_preds)
    print ('Overall:', cv)           
    return train_preds,test_preds


def cite_cos_sim_model(len_num, n_targets=140):
    #######################  svd  #######################   
    input_num = tf.keras.Input(shape=(len_num, ))     
    x = input_num
    x0 =  tf.keras.layers.Reshape((1,x.shape[1]))(x)
    x0 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1800, activation='elu', kernel_initializer='Identity',return_sequences=False))(x0)
    x1 = tf.keras.layers.GaussianDropout(0.2)(x0)         
    x2 = tf.keras.layers.Dense(1800,activation ='elu',kernel_initializer='Identity',)(x1) 
    x3 = tf.keras.layers.GaussianDropout(0.2)(x2) 
    x4 = tf.keras.layers.Dense(1800,activation ='elu',kernel_initializer='Identity',)(x3) 
    x5 = tf.keras.layers.GaussianDropout(0.2)(x4)         
    x = tf.keras.layers.Concatenate()([
                       x1,x3,x5
                      ])
    output = tf.keras.layers.Dense(n_targets, activation='linear')(x) 
    model = tf.keras.models.Model(input_num, output)
    lr=0.001
    adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-9, )
    model.compile(loss=cosine_similarity_loss, optimizer=adam,)
    model.summary()
    return model

def cite_mse_model(len_num, n_targets=140):
    
    #######################  svd  #######################   
    input_num = tf.keras.Input(shape=(len_num, ))     

    x = input_num
    x = tf.keras.layers.Dense(1500,activation ='swish',)(x)    
    x = tf.keras.layers.GaussianDropout(0.1)(x)   
    x = tf.keras.layers.Dense(1500,activation ='swish',)(x) 
    x = tf.keras.layers.GaussianDropout(0.1)(x)   
    x = tf.keras.layers.Dense(1500,activation ='swish',)(x) 
    x = tf.keras.layers.GaussianDropout(0.1)(x)    
    x =  tf.keras.layers.Reshape((1,x.shape[1]))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(700, activation='swish',return_sequences=False))(x)
    x = tf.keras.layers.GaussianDropout(0.1)(x)  
    
    output = tf.keras.layers.Dense(n_targets, activation='linear')(x) 

    model = tf.keras.models.Model(input_num, output)
    
    lr=0.0005
    weight_decay = 0.0001
    
    # Note that in the original code, tfa was used, which is no longer supported
    opt = tf.keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay
    )    

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt,)
    model.summary()
    return model


def train_nn_models(train_cell_ids, train_cite_X, train_cite_y, test_cell_ids, test_cite_X, folds, EPOCHS=100):
    """Train 2 neural network models and blend their predictions
    
    This is based on the original code from senkin13: https://www.kaggle.com/code/senkin13/2nd-place-gru-cite
    The code here is slightly modified to work with updated versions of the libraries,
    but essentially it is the same. Each model is trained 5 times on 4 folds of the training data,
    and the predictions are averaged. Validation scores for unseen fold are reported.
    The final predictions are a weighted average of predictions from the two models.

    Parameters
    ----------
    train_cell_ids : list
        List of cell IDs for training
    train_cite_X : np.ndarray
        Array of training features
    train_cite_y : np.ndarray
        Array of training targets
    test_cell_ids : list
        List of cell IDs for testing
    test_cite_X : np.ndarray
        Array of testing features
    folds : sklearn.model_selection.KFold
        KFold object for cross-validation
    EPOCHS : int
        Number of epochs to train the models

    Returns
    -------
    train_preds : np.ndarray
        Array of training predictions
    test_preds : np.ndarray
        Array of testing predictions
    """
    train_preds_cos, test_preds_cos = nn_kfold(
        train_cell_ids,
        train_cite_X,
        train_cite_y,
        test_cell_ids,
        test_cite_X,
        cite_cos_sim_model,
        folds,
        'cite_cos_model',
        BATCH_SIZE=620,
        EPOCHS=EPOCHS,
        LR_FACTOR=0.05
    )

    # zscore for target
    train_cite_y = zscore(train_cite_y)

    train_preds_mse, test_preds_mse = nn_kfold(
        train_cell_ids,
        train_cite_X,
        train_cite_y,
        test_cell_ids,
        test_cite_X,
        cite_mse_model,
        folds,
        'cite_mse_model',
        BATCH_SIZE=600,
        EPOCHS=EPOCHS,
        LR_FACTOR=0.1
    )

    train_preds_cos = zscore(train_preds_cos)
    train_preds_mse = zscore(train_preds_mse)
    train_preds = train_preds_cos * 0.55 + train_preds_mse * 0.45
    cv = correlation_score(train_cite_y,  train_preds)
    print ('Blend:', cv)     

    test_preds_cos = zscore(test_preds_cos)
    test_preds_mse = zscore(test_preds_mse)
    test_preds = test_preds_cos * 0.55 + test_preds_mse * 0.45

    return train_preds, test_preds