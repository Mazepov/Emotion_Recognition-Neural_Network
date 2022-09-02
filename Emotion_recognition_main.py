#!/usr/bin/env python
# coding: utf-8

# Разработаем детектор лица который будет работать совместно с нашей нейросетью по распознованию эмоций

# Необходимо скачать детектор по ссылке
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# In[1]:


#Импорт необходимых библиотек
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Класс по распознованию эмоций

# In[71]:


class EmotionRecognize:
    
    """
    Для корректной работы необходимо установить и экспортировать библиотеки: tensorflow, opencv, numpy, matplotlib.
    При создании экземпляра класса, необходимо указать путь до сохраненной модели.
    Необходимо указать, ускорена ли модель с помощью TensorRT (по умол. True). 
    Включить камеру можно при инициализации модели, либо вызвав camera_mode. 
    При создании экземпляра класса можно включить запись с веб-камеры, 
    для этого укажите video_rec=True (по умол. False)
    """

        
    #Список эмоций
    em_dict = {'anger':0, 
               'contempt':1, 
               'disgust':2,
               'fear':3, 
               'happy':4, 
               'neutral': 5, 
               'sad': 6, 
               'surprise':7,
               'uncertain':8}
    #Словарь с эмоциями (ключ:значение -> значение:ключ)
    em_names = {v:k for k, v in em_dict.items()} 
        
    #Заготовка под картинку с лицом
    img = np.empty((1, 48, 48, 1))
    
    
    
    def __init__(self, model_path, cam_mode = None, video_rec = False, TensorRT=True):
        self.model = tf.keras.models.load_model(model_path)
        self.cam_mode = cam_mode
        self.video_rec = video_rec
        self.TensorRT = TensorRT
        self.cam = None
        self.out = None

    
    
    def camera_mode(self, cam_mode=None):
        
        """
        Включение или отключение камеры.
        On - Включить камеры
        Off - Отключить камеру
        """

        if self.cam_mode == 'On' or cam_mode == 'On':
            self.cam = cv2.VideoCapture(0)
            self.cam_mode = 'On'
            print('Камера запущена!')
            
        elif cam_mode == 'Off':
            self.cam_mode == 'Off'
            cv2.destroyAllWindows()
            print('Камера выключена!')
            
        else:
            cv2.destroyWindow('cam')
            print('Камера НЕ подключена. Проверьте настройки.')
            

    
    def recognize_emotion(self):
        
        """
        Воспроизведение видеопотока с вебкамеры с инициализацией эмоций
        """
        
        if self.cam_mode == 'On':
            if self.cam == None:
                self.cam = cv2.VideoCapture(0)
        else:
           raise Exception('Камера НЕ подключена. Проверьте настройки.')

        # иницилизируем детектор
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        
        
        if self.video_rec == True: #Запись видео
            
            #Размер изображения с камеры
            frame_width = int(self.cam.get(3))
            frame_height = int(self.cam.get(4))
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID') #Кодек
            fps=25 #фпс
            self.out = cv2.VideoWriter('./output1.avi', fourcc, fps, (frame_width,
                                                                      frame_height))
        

        while(True):
            ret, frame = self.cam.read()
            

            #Переведем в ЧБ формат
            grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Определяем лица в кадре
            faces = face_detector.detectMultiScale(grayscale_image, 1.3, 5)


            if len(faces) != 0:  
                for i in range(len(faces)):

                    #BoundingBoxes будут двух цветов (red и green)
                    if i%2 == 0:
                        color_b = (0,0,255)
                    else:
                        color_b = (0,255,0)

                    x, y, w, h = faces[i]

                    #Вырезаем лицо из кадра
                    face_bgr = frame[y:y + h, x:x + w]
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                    #Обработка
                    #Конвертирование в Ч/Б
                    face_gray = tf.image.rgb_to_grayscale(face_rgb)

                    #Изменение размера
                    image = tf.image.resize(face_gray, (48,48)) /255.
                    img_face =  np.array(image)

                    #Конвертация в нужный формат
                    self.img[0] = img_face

                    #Определение эмоции
                    if self.TensorRT == True:
                        em_pred = self.model(self.img.astype('float32')) #Модель ускорена с помощью TensorRT
                    else:
                        em_pred = self.model.predict(self.img) #Модель НЕ ускорена
                        
                    emotion = self.em_names[np.argmax(em_pred)]

                    #BoundingBox
                    frame_with_bb = cv2.rectangle(frame, (x, y), (x + w, y + h), color_b, 3)

                    #Наносим эмоцию
                    frame_with_bb_and_emotion = cv2.putText(frame_with_bb, 
                                                            emotion, (x, y - 10), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_b, 2)
                    
                
                if self.video_rec == True:
                    self.out.write(frame_with_bb_and_emotion)

                cv2.imshow("Emotion recognition", frame_with_bb_and_emotion)
                
            else:
                if self.video_rec == True:
                    self.out.write(frame)
                    
                cv2.imshow("Emotion recognition", frame)
                


            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.video_rec == True:
                    self.cam.release()
                    self.out.release()
                break

