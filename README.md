# Emotion_Recognition-Neural_Network

Нейронная сеть по распознаванию эмоций.

Emotion_Recognition_main подключается к веб камере и с помощью каскадов Хаара (библиотека OpenCV) детектирует лица, а с помощью нейросетевой модели определяет одну из девяти возможных эмоций. Emotion_Recognition_main разработан для работы с custom архитектурой нейронной сети. При использовании другой архитектуры (например VGG19) необходимо делать препроцессинг изображения, соответствующий выбранной модели.

При разработке модели использовалось несколько архитектур. 

Наилучший результат показала модель c архитектурой VGG19, предобученная на ImageNet. Точность на валидации составила 52%.
Custom модель, с небольшим количеством сверточных слоев, показала точность на валидации 45%.

При инференсе использовалось ускоренее с помощью TensorRT, однако есть возможность отключения ускорения при создании экземпляра класса в Emotion_Recognition_main.
