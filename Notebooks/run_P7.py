# run_P7.py "dog.jpg"

import sys
import pickle
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
# 20 races : ['Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 
# 'Blenheim_spaniel', 'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 
# 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound', 
# 'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound']

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    img_path = sys.argv[1]
    print('Chargement de', img_path)
    img = load_img(img_path, target_size=(224, 224))  # Charger l'image

    img = img_to_array(img)  # Convertir en tableau numpy
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16

    # # model = VGG16()
    # model = load_model('best_model')
    # y = model.predict(img)
    # print('Cette image représente :', decode_predictions(y, top=1)[0][0][1])

    base_model = load_model('base_model.h5')
    Xvgg = base_model.predict(img)
    Xvgg_reshape = Xvgg.reshape(Xvgg.shape[0], -1)
    sc = load_obj('sc')
    X = sc.transform(Xvgg_reshape)
    svc = load_obj('svc')
    y_pred = svc.predict(X)

    lbl_enc = load_obj('lbl_encoder')
    y_lib = lbl_enc.inverse_transform(y_pred)
    print('Cette image représente un', y_lib[0])
