#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:powderblue;"><h1>Projet Synthese</h1><b><br>
#     Groupe 3: Amardeepkumar Haulkhory, Hilaire Yuma, Jian-Guo Zhao, Mamadou Sy</b></div>

# <h2>Partie II Apprentissage par <abbr title="Convolutional Neural Networks"><b>CNN </b></abbr></h2>
# <p>Definir le <abbr title="Convolutional Neural Networks"><b>CNN </b></abbr>architecture:</p>
# <ol>
#     <li>La premiere couche, Conv2D, prend un batch de données avec input_shape=(48, 48, 1)</li>
#     <li>La deuxieme couche, Conv2D, prend la sortie de couche 1, keras determine le shape automatiquement</li>
#     <li>Il n'est pas necessaire d'indiquer input_shape pour les couches plus profondes</li>
#     <li>La troisieme couche, Conv2D, prend la sortie de la couche 2, keras determine le shape automatiquement</li>
#     <li>Ainsi de suite, on peut ajouter autres couches Conv2D s'il est nécessaire</li>
#     <li>Entre chaque deux couches Conv2D, une couche Max pooling est ajoutée</li>
#     <li>Pour reduire le sur-apprentissage, une couche de regularisation Dropout pourrait etre ajoutée</li>
#     <li>Pour obtension d'une meilleure performance, une couche Normalisation pourrait etre aussi appliquée</li>
#     <li>Avant la couche dense, nous ajoutons une couche Flatten qui convert le matrix 2D en vector 1D</li>
#     <li>Puis, une couche Dense avec une fonction activation ReLu</li>
#     <li>Encore une couche de regularization Dropout est appliquée avec une valeur plus grande normallement</li>
#     <li>Enfin, la couche de sortie donne 7 neurons pour les 7 classes avec une fonction activation softmax qui donne la possibilite de prediction pour chaque class.</li>
# </ol>

# <h3>Dépendences - packages nécessaires</h3>
# <p>En utilisant <b>tensorflow.compat.v1</b>, vous devez desactiver eager_execution.<br>Ce n'est pas le cas pour <b>tensorflow.v2</b>. De plus, le package <b>tensorflow</b> n'a pas la fonction <b><i>disable_eager_execution</i></b>.</p>

# In[1]:


import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D 
from keras.utils import to_categorical

from keras.utils.data_utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator

import numpy as np      # linear algebra
import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import os
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from numpy import expand_dims
from keras.preprocessing.image import img_to_array

get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>Chargement des donnees</h3>
# <p>Pour eviter l'operation <b>download</b> des donnees se fait chaque fois, on les sauvegarde dans un repertoire local, qui est differente pour different colaborateurs. Veuillez changez la valeur de variable <b><i>data_dir</i></b> si necessaire</p>

# In[2]:


data_dir = "C:\\Users\\jeang\\Documents\\BDEB\\A62\\ProSyn3\\data\\fer2013.csv"
df = pd.read_csv(data_dir )


# <h3>Repartition des donnees</h3>
# <p>Les donnees originales sont repartie en trois partie. Selons Partie I, la cible est bien balance dans ces trois parties</p>

# In[3]:


df_training = df[df['Usage']=='Training']
df_validation = df[df['Usage']=='PublicTest']
df_test = df[df['Usage']=='PrivateTest']


# <h3>Affichage des donnees</h3>
# <p> Ces fonctions affichent un image (une ligne)<br>Notons que 48 x 48 = 2304</p>

# In[4]:


FER2013_WIDTH = 48
FER2013_HEIGHT = 48

# indices 0 - 6 correspondent emotions suivantes
Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  

def string_to_image(str_in):
    return np.reshape(str_in.split(" "), (FER2013_WIDTH, FER2013_HEIGHT)).astype("float")

def fer2013_show_instance(index, d=df):
    """Shows the image and the emotion label of the index's instance."""
    category, str_image = d.at[index,'emotion'], d.at[index,'pixels']
    image = string_to_image(str_image)
    plt.imshow(image, cmap="gray")
    plt.title(f"Exemple de categorie {category}: {Emotions[category]}")
    
fer2013_show_instance(np.random.randint(0,len(df_training)))


# <h3>Transformation des données</h3>
# <p><ul> La fonction <b><i>image_to_4d_array</i></b> prend comme entree une dataframe qui tienne les donnees d'images en format des chaines de caracteres.<br>Elle retoune comme sortie un <b>np.array</b> de <mark>4 dimensions</mark>, normalisee, x, et categoriel y, utilisables comme entree pour <b>tensorflow CNN<b></ul></p>

# In[5]:


def image_to_4d_array(d=df, nclass=7):
    """Transforms the (blank separated) pixel strings in the DataFrame to an 4-dimensional array 
    (1st dim: instances, 2nd and 3rd dims represent 2D image, 4th dim represent the color)."""
    
    CLASS_NUMBER = min(nclass, 7)
    
    print("Converting pixels to 2D image ...")
    pixels_list = d["pixels"].values
    list_image_2d = [string_to_image(pixels) for pixels in pixels_list]
        
    # Convert list to 4D array:
    X = np.expand_dims(np.array(list_image_2d), -1)
    X = X / 255  
    
    print("Converting emotion to categorical n-array ...")
    Y = to_categorical(d.emotion, CLASS_NUMBER)
    
    print(f"The given dataset has been converted to {X.shape} array")
    
    return X, Y


# <p>On applique la fonction <b><i>image_to_4d_array</i></b> sur les trois partitions</P>

# In[6]:


X_train, Y_train = image_to_4d_array(df_training)
X_test, Y_test = image_to_4d_array(df_test)
X_valid, Y_valid = image_to_4d_array(df_validation)


# <h3>Fonction <i>get_model_v23</i></h3>
# <p><ul>
# <li>Cette fonction <b style="backgroud-color:powerblue"><i>get_model_v23</i></b> peut prendre seulement quelques hyperparametres</li>
# <li>Elle appelle <i style="color:blue">keras.models.Sequential</i> qui nous permet de construire un modele cnn</li>
# <li>Les valeurs par defaut sont notre choix entre les meilleures, mais elle laisse des chances pour utilisateur de les choisir</li>
# <li>Elle retourne un architecture d'un modele Sequential qu'on peut l'entrainner et le tester</li>
# </ul></p>

# <h4>Points essentiels de la fonction <i>get_model_v23</i></h4>
# <p><ol>
# <li>Nombre de couche est 5 par defaut, mais elle peut etre un entier de votre choix</li>
# <li>Les nombres de filtre doivent en rapport avec le nombre de couche, preferable plus elevé pour les couches profondes.</li>
# <li>Les kernel sizes doivent en rapport avec le nombre de couche, preferable impaire et le plus petit que possible</li>
# <li>Les pooling size doivent en rapport avec le nombre de couche, preferablement petit ou zero </li>
# <li>Les dropout values doivent en rapport avec le nombre de couche aussi</li>
# </ul></p>

# In[7]:


def get_model_v23(num_layers=5, lf=256, ld=0.5,
                  num_filtre=(16, 32, 64, 128, 256), 
                  k_size=(3,3,3,3,3), 
                  p_size=(2,2,2,2,0), 
                  dropout=(0.10, 0.10, 0.15, 0.10, 0.10)):
    
    input_shape, num_classes, last_features, last_dropout = (48, 48, 1), 7, lf, ld
    
    model = Sequential()
    
    for layer in range(num_layers): 
        if layer == 0:
            model.add(Conv2D(num_filtre[layer], kernel_size=k_size[layer], activation=tf.nn.relu, padding="same", input_shape=input_shape))
        else:
            model.add(Conv2D(num_filtre[layer], kernel_size=k_size[layer], activation=tf.nn.relu, padding="same"))
            # normalization for each layer but not the first
            model.add(BatchNormalization())
        
        # max pooling with the given choose
        if p_size[layer] > 0:
            model.add(MaxPooling2D(pool_size=(p_size[layer], p_size[layer])))
        
        # Dropout with the given choose
        if dropout[layer] > 0:
            model.add(Dropout(dropout[layer]))           
        

    model.add(Flatten())                          
    model.add(Dense(last_features, activation=tf.nn.relu))
    model.add(Dropout(last_dropout))
    model.add(Dense(num_classes, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# <h3>Fonction <i>save_model</i></h3>
# <p><ul>
# <li>Cette fonction <b style="backgroud-color:powerblue"><i>save_model</i></b> prend un Sequential model et nom de fichier</li>
# <li>Elle appelle <i style="color:blue">keras.models.Sequential.to_json</i> et <i style="color:blue">save_weights</i></li>
# </ul></p>

# In[8]:


def save_model(model, file_json="", file_weight="", workdir=""):
    # serialize model to JSON
    model_json = model.to_json()
    projet_dir = workdir if len(workdir)>0 else "C:\\Users\\jeang\\Documents\\BDEB\\A62\\ProSyn3"
    model_filename = 'fer.json' if file_json=="" else file_json
    with open(os.path.join(projet_dir, model_filename), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    # model.save_weights("C:\\Users\\Nisha Haulkhory\\projet de synthese\\GIT\\ProSyn3\\fer.h5")
    weight_filename = 'fer.h5' if file_weight=="" else file_weight
    model.save_weights(os.path.join(projet_dir, weight_filename))
    print("Saved model to disk done")


# <h3>Fonction <i>show_confusion_matrix</i></h3>
# <p><ul>
#     <li>Cette fonction <b style="backgroud-color:powerblue"><i>show_confusion_matrix</i></b> prend comme entree un model</li>
#     <li>Elle appelle <i style="color:blue">model.predict</i> et le testset pour y_pred</li>
#     <li>Elle compare y_pred avec les valeurs real pour construire la matrix de confusion</li>
#     <li>Elle normalize la matrix de confusion car les cibles dans notre données originales n'est pas uniforme </li>
#     <li>Mais ceci pourrait déclencher d'erreur pour les classes manquantes car Div_by_zero donne NA</li>
# </ul></p>

# In[9]:


def show_confusion_matrix(model, x_test=X_test, y=df_test.emotion):
    submission = pd.DataFrame(model.predict(x_test))
    submission['label'] = submission.idxmax(axis=1)
    cm = confusion_matrix(submission['label'], y)               # df_test['emotion'])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm.round(2)
    return pd.DataFrame(cm)


# <h3>Class <i>BalancedDataGenerator</i></h3>
# <p><ul>
# <li>Cette class <b style="backgroud-color:powerblue"><i>BalancedDataGenerator</i></b> est trouve sur web pour imbalanced dataset</li>
# <li>Elle select non uniformement les echantillons pour les agumenter et finir avec relativement uniform dataset</li>
# </ul></p>

# In[25]:


# Je n'ai pas bien compris cette class encore
# Le résultat est aussi hors ce qui prédire dans le web
class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        datagen.fit(x)  
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, 
                                                                  sampler=RandomOverSampler(), 
                                                                  batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()


# <h3>Objet <i>ImageDataGenerator</i></h3>
# <p><ul>
# <li>Cette instance d'objet <b style="backgroud-color:powerblue"><i>datagen</i></b> est utilisable dans plusieurs cas pour data augmentation</li>
# </ul></p>

# In[11]:


datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=20,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             shear_range=0.15,
                             zoom_range=0.15,
                             horizontal_flip=True,
                             vertical_flip=False)

# bgen = BalancedDataGenerator(X_train, Y_train, datagen, batch_size=256)
# steps_per_epoch = bgen.steps_per_epoch


# <h3>Fonction <i>uniform_data_aug_fit</i></h3>
# <p><ul>
# <li>Cette fonction <b style="backgroud-color:powerblue"><i>uniform_data_aug_fit</i></b> prend comme entrees:
# <ol><li>un model a aprendre</li>
# <li>un instance ImageDataGenerator</li>
# <li>le train set et la validation set</li>
# <li>et la taille de batch bs</li></ol></li>
# <li>Elle calcule le nombre de <i style="color:blue">steps_per_epoch</i> pour controler le trade-off temps et performance</li>
# <li>La difference est qu'elle n'utilise pas le meme taux de augmentation, mias plus grand pour les classes moins populaires</li>
# <li>Elle appelle la fonction <i style="color:blue">model.fit_generator</i> pour entrainer le modele</li>
# </ul></p>

# In[12]:


def uniform_data_aug_fit(model, dgen, x_train, y_train, valid, bs, taux=9, ep=10):
    taux_augmented_data = min(taux, 48)  
    batch_size = min(bs, x_train.shape[0])
    steps_per_epoch = int(x_train.shape[0] * taux_augmented_data / batch_size)
    l = model.fit_generator(dgen.flow(x_train, y_train, batch_size=batch_size), epochs=ep, validation_data=valid,
                            verbose = 1, steps_per_epoch = steps_per_epoch)
    return model, l


# <h3>Fonction <i>balance_fit</i></h3>
# <p><ul>
# <li>Cette fonction <b style="backgroud-color:powerblue"><i>balance_fit</i></b> prend un model et un instance ImageDataGenerator</li>
# <li>Elle appelle la fonction <i style="color:blue">uniform_data_aug_fit</i> pour faire apprendre le model</li>
# <li>La difference est qu'elle n'utilise pas le meme taux de augmentation, plus grand pour les classes moins populaires</li>
# <li>En effet, elle entraine 3 fois le modele avec 3 train set differentes </li>
# </ul></p>

# In[26]:


def balance_fit(model, dgen, valid=(X_valid, Y_valid), bs=32, ep=10, fnm='fer.json', fnw='fer.h5'):
    
    #train the model with uniform_data_aug_fit but the trainset without the most popular calss 3
    train_o = df_training[(df_training.emotion != 3)]
    valid_o = df_validation[(df_validation.emotion != 3)]
    test_o = df_test[(df_test.emotion != 3)]
    
    x, y = image_to_4d_array(train_o)
    validation = image_to_4d_array(valid_o)
    x_test, y_test = image_to_4d_array(test_o)
    
    model, l = uniform_data_aug_fit(model, dgen, x, y, validation, bs, taux=12, ep=ep)
    show_confusion_matrix(model, x_test=x_test, y=test_o.emotion)
    sns.lineplot(data=pd.DataFrame(l.history)[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
    print("Evaluating the model on test data ...")
    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)
    
    # second train the model with all train set, validation set and test set without data augmentation
    l = model.fit(x=X_train, y=Y_train, validation_data=valid, batch_size=256, epochs=ep)
        
    return l 


# <h3>Fonction <i>show_save_trained_model</i></h3>
# <p><ul>
# <li>Apres avoir entraine un modele, nous voulons souvent savoir l'histoire d'apprentissage, la performance, et sauvegarder le modele</li>
# <li>Cette fonction <b style="backgroud-color:powerblue"><i>show_save_trained_model</i></b> donne une flexibilite pour different methode d'apprentissage</li>
# <li>Elle prend le model et histoire d'apprentissage comme entrees</li>
# <li>Elle compare avec les test set pour evaluer le model</li>
# <li>Elle imprine aussi la matrix de confusion </li>
# </ul></p>

# In[14]:


def show_save_trained_model(model, history, fnm, fnw, x_test=X_test, y_test=Y_test, y=df_test.emotion):
    sns.lineplot(data=pd.DataFrame(history.history)[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
    save_model(model, file_json=fnm, file_weight=fnw)
    print("Evaluating the model on test data ...")
    results = model.evaluate(x_test, y_test)
    print("test loss, test acc:", results)
    print(show_confusion_matrix(model, x_test, y))


# <h3>Fonction <i>fit_test_save_model</i></h3>
# <p><ul>
# <li>Cette fonction <b style="backgroud-color:powerblue"><i>fit_test_save_model</i></b> regroupe plusieurs commandes</li>
# <li>Elle fournie une interface pour l'apprentissage, evaluation, visualisation, et sauvegarde du model cnn</li>
# <li>Si le drapeau <b style="backgroud-color:powerblue"><i>aug</i></b> est False, elle apprend le model avec trainset et validation set</li>
# <li>Si le drapeau <b style="backgroud-color:powerblue"><i>aug</i></b> est True, elle apprend le model avec augmented trainset</li>
# <li>Une fois l'apprentissage se termine, elle execute des commandes pour display l'histoire d'apprentissage, sauvegarder le model etc</li>
# </ul></p>

# In[27]:


def fit_test_save_model(model, x_train=X_train, y_train=Y_train, valid=(X_valid, Y_valid), 
                        aug=True, imbalance=False, dgen = datagen,
                        bs=32, ep=10, fnm='fer.json', fnw='fer.h5'):
    
    print("Fitting the model ...")
    batch_size = min(bs, x_train.shape[0])
    
    if not aug:
        l = model.fit(x=x_train, y=y_train, validation_data=valid, batch_size=batch_size, epochs=ep)
        
    elif imbalance:
        # I prefer balance_fit because I know how it works
        l = balance_fit(model, dgen, valid, bs=batch_size, ep=ep)
        
        # I tried this method but I did not get reasonable result: There are something wrong
        # bgen = BalancedDataGenerator(x_train, y_train, dgen, batch_size=batch_size)
        # steps_per_epoch = bgen.steps_per_epoch
        # l = model.fit_generator(bgen, steps_per_epoch, epochs=ep, validation_data=valid, verbose = 1)
    
    else:
        _, l = uniform_data_aug_fit(model, dgen, x_train, y_train, taux=9, bs=batch_size, ep=ep, valid=valid)

    show_save_trained_model(model, l, fnm, fnw)
    
    return l


# <h3>1er exemple de construction de CNN model</h3>
# <p>Par defaut, il y a 5 couches conv2D, la taille k_size est (3,3) pour chaque couche</p>
# <p>Notons que les valeurs dropout sont 0.1 pour ces 5 couches, mais last_dropout(ld) est 0.45</p>

# In[30]:


#Same architecture will used for 3nd eample
model = get_model_v23(num_filtre=(16, 32, 64, 128, 256), lf=256, ld=0.45, p_size=(2,2,2,2,0), dropout=(0.1, 0.1, 0.1, 0.1, 0.1))
model.summary()


# In[31]:


l = fit_test_save_model(model, bs=32, ep=25, fnm='fer238.json', fnw='fer238.h5')


# <h4>Commentaires</h4>
# <p>Tand qu'il n'y a pas de overfitting, un modele peut etre continuellement entrainne jusqu'a la performance desire. Ceci est vrai au moins sur le même training set.<br> Nous constatons, accuracy s'améliore dans chaque epoch, et val_accuracy a aussi unetendance d'augmenter. Dans chaque epoch, il y a 8074 steps, ceci est definie par la taille de notre training set, le taux d'augmentation, et la batch_size (28709*9/32).<br> Notons que ce nombre de steps doit etre assez eleve pour un meilleur apprentssage. Mais le temps d'apprentissage augmente aussi rapidement avec ce nombre. Pour rouler la commande ci-dessus, mon laptop a travaille forte pendant environ 14000 secondes, ou 3.89 heurs.</p>
# <p>Effet nombre filtres: forte nombre filtre est preferable pour chercher features, mais non favorable pour overfitting. Non plus pour le temps d'apprentissage. Le plus important, mettre le nombre de filtre plus elevé dans les couches profondes. Ceci est favorable pour réduire le temps d'apprentissage.</p>
# <p>Effet dropout: forte dropout est preferable pour controler overfitting, mais non favorable pour ralentir la vitesse d'apprentissage. C'est un trade-off dans l'apprentissage par CNN</p>
# <p>Meilleur performance obtenu au test set avant était 64.5% lorsque dropout=(0.1,0.1,0.1,0.1,0.1), ld=0.35, ep=10. Avec les condition ci-dessus, nous avons eu de encore meilleure performance: 66%</p>

# <h3>Proposition nouveau essai</h3>
# <p>Pouvons-nous entrainer un modele avec different train set? Si l'apprentissage du modele s'ameliore continuellement malgre le changement de train set, nous pouvons traiter le probleme imbalance de notre train set. En effet, la class la plus populaire happy(3) a une performance environ 91% mais la class sad(4) a seulement 44% selon un matrix de confusion.<br>Si, au lieu d'augmenter tous train set 9 fois, on augmente seulement 12 fois les autre classes moins populaires, on pourrait sauve du temps d'apprentissage.<br>
# En basant sur cette proposition, nous allons traiter notre defi d'over-fitting et unbalanced dataset ci-dessous.<br>En 3eme exemple, nous allons reprendre cette technique.</p>

# In[39]:


model = get_model_v23(num_filtre=(16, 32, 64, 128, 256), lf=256, ld=0.5, p_size=(2,2,2,2,0), dropout=(0.1, 0.1, 0.1, 0.1, 0.1))
model.summary()


# In[40]:


train_o = df_training[(df_training.emotion != 3)]
valid_o = df_validation[(df_validation.emotion != 3)]
test_o = df_test[(df_test.emotion != 3)]
    
x, y = image_to_4d_array(train_o)
validation = image_to_4d_array(valid_o)
x_test, y_test = image_to_4d_array(test_o)
    
model, l = uniform_data_aug_fit(model, datagen, x, y, valid=validation, bs=32, taux=9, ep=10)
show_confusion_matrix(model, x_test=x_test, y=test_o.emotion)
sns.lineplot(data=pd.DataFrame(l.history)[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
print("Evaluating the model on test data ...")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)


# In[44]:


# Pour corriger les erreur ci-haute
show_confusion_matrix(model, x_test=x_test, y=test_o.emotion)


# In[45]:


# Pour corriger les erreur ci-haute
sns.lineplot(data=pd.DataFrame(l.history)[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);


# In[46]:


# Pour corriger les erreur ci-haute
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)


# In[47]:


# Not sure if these data are overwriten
#x, y = image_to_4d_array(df_training)
#validation = image_to_4d_array(df_validation)
#x_test, y_test = image_to_4d_array(df_test)

#X_train, Y_train = image_to_4d_array(df_training)
#X_test, Y_test = image_to_4d_array(df_test)
#X_valid, Y_valid = image_to_4d_array(df_validation)

l = model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), batch_size=256, epochs=30)    
#model, l = uniform_data_aug_fit(model, datagen, x, y, valid=validation, bs=32, taux=9, ep=10)
show_confusion_matrix(model, x_test=X_test, y=df_test.emotion)
sns.lineplot(data=pd.DataFrame(l.history)[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);
print("Evaluating the model on test data ...")
results = model.evaluate(X_test, Y_test)
print("test loss, test acc:", results)


# In[48]:


show_confusion_matrix(model, x_test=X_test, y=df_test.emotion)


# <h4>Commentaires sur 2eme apprentissage avec different train set</h4>
# <p>Eventuellement, le changement de data-set pour 2eme apprentissage d'un modele semble etre une technique possitive. Pour apprendre d'un modele vierge par uniform_data_augmentation, on a besoin environ 14000 (570x25) seconds pour une performance de 64%<br>
# Avec un apprentissage par uniform_data_augmentation sur les autres classes que la plus populaire puis un 2eme apprentissage sans data_augmentation, on a besoin environ 5900 (430x10+54X30) seconds pour une performance simulaire. Un économie de plus que 50% .</p>

# <h3>2eme exemple de construction de CNN model</h3>
# <p>Par defaut, il y a 5 couches, la taille k_size est (3,3) pour chaque couche</p>
# <p>Notons que les valeurs dropout sont 0.25 pour ces 5 couches, et last_dropout(ld) est 0.5</p>

# In[49]:


model = get_model_v23(num_filtre=(16, 32, 64, 128, 256), lf=256, ld=0.5, p_size=(2,2,2,2,0), dropout=(0.25, 0.25, 0.25, 0.25, 0.25))
model.summary()


# <h3>2eme Exemple d'apprendre d'un CNN model</h3>
# <p>Cette fois, pas de technique de data augmentation est utilisee</p>

# In[50]:


l = fit_test_save_model(model, bs=256, ep=50, aug=False, fnm='fer234.json', fnw='fer234.h5')


# <h4>Commentaires</h4>
# <p>Nous observons d'over-fitting malgré forte dropout. Une performance de 62.7% est obtenue sans data augmentation. Le temps d'apprentissage est relativement courte: environ 2700s (54x50)</p>

# <h3>3eme Exemple d'apprendre un CNN model</h3>
# <p>Cette fois, nous avons utilisé une technique de deux entrainements sequentiels avec imbalanced data augmentation pendant le premier entrainement seulement</p>

# In[28]:


model = get_model_v23(num_filtre=(16, 32, 64, 128, 256), lf=256, ld=0.45, p_size=(2,2,2,2,0), dropout=(0.1, 0.1, 0.1, 0.1, 0.1))
model.summary()


# In[29]:


l = fit_test_save_model(model, bs=32, ep=10, imbalance=True, fnm='fer235.json', fnw='fer235.h5')


# <h4>Commentaires</h4>
# <p>Nous observons d'over-fitting seulement pour la deuxieme entrainement malgré faible dropout. Une performance de 63.9% est obtenue qui est de même que uniform data augmentation. Le temps d'apprentissage est relativement courte, environ 6000s (555+51)x10. </p>

# <h2>Conclusion</h2>
# <p><ul>
# <li>L'interface de construction du model cnn et de l'apprentissage devient simplement deux appels de deux fonctions.</li>
# <li>Par choix des hyperparametres, on peut avoir differents modeles et la technique d'apprentissage.</li>
# <li>Sans data augmentation, maximum accuracy sur test set est 62% en ce moment. Ce n'est pas le plus haute.</li>
# <li>De plus, sans data augmentation, forte dropout sont nécessaires pour lutter contre le sur-apprentissage.</li>
# <li>Avec data augmentation, nous obtenous de meilleurs performance: 66% dans le cas keras.ImageDataGenerator.</li>
# <li>Avec cette technique, faible valeurs dropout sont suffit pour empecher overfitting.</li>
# <li>mais le temps d'apprentissage est relativement longue.</li>
# <li>Pour reduire le temps d'apprentissage, nous avons utilise une technique telle que seulement les classes qui sont difficiles à classifier ont eu une data augmentation, le modele est ensuite entrainé par la même technique sans data augmentation. Nous avons eu une performance aussi elevée que 64% mais le temps d'apprentissage est réduit à plus que 50% (6060/14600)</li>
#     <li>Avec une technique oversampling, Nous n'avons pas eu de bon résultat, faut du temps serré et manque de compréhension de cette technique. Voir les cellules qui suivent la conclusion</li>
# </ul></p>

# <h2>Annexe</h2>
# <p>Note sur une mauvaise expérience avec une class <b><i style="color:blue">BalancedDataGenerator</i></b></p>

# <h3>3eme exemple de construction du CNN model</h3>
# <p>Par defaut, il y a 5 couches, la taille k_size est (3,3) pour chaque couche</p>
# <p>Notons que les valeurs dropout sont 0.1 pour les 5 couches, mais last_dropout(ld) est 0.45. Cette architecture a ete utilisee pour la premier exemple</p>

# In[21]:


model = get_model_v23(num_filtre=(16, 32, 64, 128, 256), lf=256, ld=0.45, p_size=(2,2,2,2,0), dropout=(0.1, 0.1, 0.1, 0.1, 0.1))
model.summary()


# <h3>3eme Exemple d'apprendre un CNN model</h3>
# <p>Cette fois, la technique de imbalanced data augmentation est utilisee</p>

# In[24]:


l = fit_test_save_model(model, bs=32, ep=10, imbalance=True, fnm='fer235.json', fnw='fer235.h5')


# In[45]:


datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=45,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=.2,
                             horizontal_flip=True,
                             vertical_flip=True)

#bgen = BalancedDataGenerator(X_train, Y_train, datagen, batch_size=256)
#steps_per_epoch = bgen.steps_per_epoch


# In[ ]:





# In[ ]:





# In[ ]:





# <h2> References </h2>
# <p>keras image data generator class <u style="color:blue"><i> https://keras.io/api/preprocessing/image/</i></u><br>Ce que c'est <b>"Imbalanced Data"</b>: <u style="color:blue"><i>https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb</i></u><br>
# Comment traiter <b>"Imbalanced image data"</b>: <u style="color:blue"><i>https://medium.com/analytics-vidhya/how-to-apply-data-augmentation-to-deal-with-unbalanced-datasets-in-20-lines-of-code-ada8521320c9</i></u><br>
# Code original pour <b>"Imbalanced image data"</b>: <u style="color:blue"><i>https://github.com/scikit-learn-contrib/imbalanced-learn</i></u></p>
# <p>By using ImageDataGenerator in Keras, we make additional training data for data augmentation. Therefore, the number of samples for training can be set by yourself. If you want two times training data, just set steps_per_epoch as (original_sample_size*2)/batch_size.<u style="color:blue"><i>https://stackoverflow.com/questions/47928347/value-of-steps-per-epoch-passed-to-keras-fit-generator-function</i></u></p>

# In[ ]:




