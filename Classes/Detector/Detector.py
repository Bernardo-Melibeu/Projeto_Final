# %%
import matplotlib.pyplot as plt
import mido
from keras.models import load_model
import numpy as np
from drawer import mid2arry
from keras.preprocessing import image

# %%
MODEL_ADDRESS="C:/Users/Pichau/Desktop/Nova_Pasta/Nova_Pasta/Learn_machine_learn/neura/models/4camadas54batch2xneuronios.h5"
HEIGHT = 320
WIDTH = 240

# %%
class Detector:
    def __init__(self,filename):
        img=self.inicializa(filename)
        if img!=-1:
            pred=self.aplica_model(img)
            self.faz_prompt(pred)

    def inicializa(self,filename):
        try:
            mid=mido.MidiFile(filename)
            result_array = mid2arry(mid)
            _=plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 107)), marker='.', markersize=1, linestyle='')
            _=plt.savefig("img/temp.png")
            return "img/temp.png"
        except:
            print("Erro na leitura do arquivo.")
            return -1
        
    def aplica_model(self,img):
        model=load_model(MODEL_ADDRESS)
        img = image.load_img(img, target_size = (WIDTH, HEIGHT))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        pred=model.predict(img)
        return np.argmax(pred)
    
    def faz_prompt(self,pred):
        dic={0:"Country",1:"Hip-Hop",2:"Jazz",3:"Reggae",4:"Rock"}
        print(f"Genero da musica: {dic[pred]}")


