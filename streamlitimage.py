from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as stream


stream.title('Загрузите картинку')
url = stream.text_input("Введите ссылку:")


if url:
    try:
        image = io.imread(url)
        img = image[:, :, 0]
        img = np.float32(img)
        U, sing_values, V = np.linalg.svd(img)
        sigma = np.zeros_like(img)

        np.fill_diagonal(sigma, sing_values)
        top_k = stream.sidebar.slider('Регулируйте значение для более лучшей детализации', 1,len(sing_values),5)

        trunc_U = U[:, :top_k]
        trunc_sigma = sigma[:top_k, :top_k]
        trunc_V = V[:top_k, :]
        trunc_img = trunc_U@trunc_sigma@trunc_V
        trunc_img = (trunc_img - np.min(trunc_img)) / (np.max(trunc_img) - np.min(trunc_img))
        stream.image(trunc_img)
    except Exception as e:
        stream.write(f"Произошла ошибка при загрузке изображения: {str(e)}")

