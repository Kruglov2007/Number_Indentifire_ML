import os
import random
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    file_name = "mnist.npz"
    url = "https://googleapis.com"
    
    if not os.path.exists(file_name):
        print("Загрузка датасета MNIST из интернета...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_name, 'wb') as out_file:
            out_file.write(response.read())
        print("Загрузка успешно завершена!")

    with np.load(file_name) as f:
        x_train = f['x_train'].astype("float32") / 255
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        y_train = f['y_train']
        y_train = np.eye(10)[y_train]
        
    return x_train[:50000], y_train[:50000], x_train[50000:], y_train[50000:]

train_images, train_labels, val_images, val_labels = load_dataset()

weights_file = "weights_deep.npz"

# Загрузка или инициализация весов для ТРЁХ слоев
if os.path.exists(weights_file):
    print("Найдены сохраненные глубокие веса! Загрузка модели...")
    weights = np.load(weights_file)
    w_ih1 = weights['w_ih1']
    w_h1h2 = weights['w_h1h2']
    w_h2o = weights['w_h2o']
    b_ih1 = weights['b_ih1']
    b_h1h2 = weights['b_h1h2']
    b_h2o = weights['b_h2o']
    skip_training = True
else:
    print("Сохраненных весов не найдено. Инициализация глубокой модели...")
    # Размеры слоев: 784 -> 64 -> 32 -> 10
    w_ih1 = np.random.uniform(-0.3, 0.3, (64, 784))
    w_h1h2 = np.random.uniform(-0.3, 0.3, (32, 64))
    w_h2o = np.random.uniform(-0.3, 0.3, (10, 32))
    
    b_ih1 = np.zeros((64, 1))
    b_h1h2 = np.zeros((32, 1))
    b_h2o = np.zeros((10, 1))
    skip_training = False

epochs = 5
learning_rate = 0.03  # Чуть увеличили шаг, так как слоев стало больше
l2_lambda = 0.0001

if not skip_training:
    for epoch in range(epochs):
        print(f"Epoch №{epoch}")
        train_loss = 0
        train_correct = 0
        
        # --- ОБУЧЕНИЕ ---
        for image, label in zip(train_images, train_labels):
            image = np.reshape(image, (-1, 1))   
            label = np.reshape(label, (-1, 1))   
            
            # 1. Прямой проход (Forward)
            h1_raw = b_ih1 + w_ih1 @ image
            h1 = 1 / (1 + np.exp(-h1_raw))  # Выход 1-го скрытого слоя
            
            h2_raw = b_h1h2 + w_h1h2 @ h1
            h2 = 1 / (1 + np.exp(-h2_raw))  # Выход 2-го скрытого слоя
            
            output_raw = b_h2o + w_h2o @ h2
            output = 1 / (1 + np.exp(-output_raw))  # Выход сети
            
            train_loss += np.sum((output - label) ** 2)
            train_correct += int(np.argmax(output) == np.argmax(label))
            
            # 2. Обратное распространение (Backpropagation)
            # Выходной слой -> Скрытый 2
            delta_output = output - label
            w_h2o += -learning_rate * (delta_output @ np.transpose(h2) + l2_lambda * w_h2o)
            b_h2o += -learning_rate * delta_output
            
            # Скрытый 2 -> Скрытый 1
            delta_h2 = (np.transpose(w_h2o) @ delta_output) * (h2 * (1 - h2))
            w_h1h2 += -learning_rate * (delta_h2 @ np.transpose(h1) + l2_lambda * w_h1h2)
            b_h1h2 += -learning_rate * delta_h2
            
            # Скрытый 1 -> Входной слой
            delta_h1 = (np.transpose(w_h1h2) @ delta_h2) * (h1 * (1 - h1))
            w_ih1 += -learning_rate * (delta_h1 @ np.transpose(image) + l2_lambda * w_ih1)
            b_ih1 += -learning_rate * delta_h1
            
        # --- ВАЛИДАЦИЯ ---
        val_correct = 0
        for image, label in zip(val_images, val_labels):
            image = np.reshape(image, (-1, 1))
            h1 = 1 / (1 + np.exp(-(b_ih1 + w_ih1 @ image)))
            h2 = 1 / (1 + np.exp(-(b_h1h2 + w_h1h2 @ h1)))
            output = 1 / (1 + np.exp(-(b_h2o + w_h2o @ h2)))
            val_correct += int(np.argmax(output) == np.argmax(label))
            
        print(f"Train Loss: {round((train_loss / train_images.shape[0]), 5)}")
        print(f"Train Accuracy: {round((train_correct / train_images.shape[0]) * 100, 3)}%")
        print(f"Validation Accuracy: {round((val_correct / val_images.shape[0]) * 100, 3)}%\n")

    # Сохранение усложненной структуры весов
    np.savez(weights_file, w_ih1=w_ih1, w_h1h2=w_h1h2, w_h2o=w_h2o, b_ih1=b_ih1, b_h1h2=b_h1h2, b_h2o=b_h2o)
    print(f"Обучение глубокой модели завершено. Веса сохранены в '{weights_file}'.")

# --- ТЕСТИРОВАНИЕ ---
test_image = random.choice(val_images)
test_image_input = np.reshape(test_image, (-1, 1))

h1 = 1 / (1 + np.exp(-(b_ih1 + w_ih1 @ test_image_input)))
h2 = 1 / (1 + np.exp(-(b_h1h2 + w_h1h2 @ h1)))
output = 1 / (1 + np.exp(-(b_h2o + w_h2o @ h2)))

plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"Deep NN suggests the number is: {output.argmax()}")
plt.show()
