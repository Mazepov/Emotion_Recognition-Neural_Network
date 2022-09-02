# Emotion_Recognition-Neural_Network

Нейронная сеть по распознаванию эмоций.

Emotion_Recognition_main подключается к веб камере и с помощью каскадов Хаара (библиотека OpenCV) детектирует лица, а с помощью нейросетевой модели определяет одну из девяти возможных эмоций.

При разработке модели использовалось несколько архитектур. 
В данный момент, наилучший результат показала кастомная модель, с небольшим количеством сверточных слоев (точность на валидации 44%).
Архитектура VGG16 с весами датасета "ImageNet" показала точно на валидации 14%, таким образом не имеет смысла использовать эту архитектуру.
В настоящий момент тестируется модель с архитектурой обученной на датасета VGGFace. Время обучения одной эпохи составляет практически 5 часов, обучение проходит долго. Однако уже на 3 эпохе точность на валидации составляет 38%. Модель будет обучаться до тех пор пока точность не выйдет на плато.

При инференсе использовалось ускоренее с помощью TensorRT, однако есть возможность отключение ускорение при создании экземпляра класса в Emotion_Recognition_main.

