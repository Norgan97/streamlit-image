from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as stream
from sklearn.preprocessing import StandardScaler


stream.title('Загрузите картинку')
url = stream.text_input("Введите ссылку:")


if url:
    try:
        image = io.imread(url)
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
        red_channel = np.float32(red_channel)
        green_channel = np.float32(green_channel)
        blue_channel = np.float32(blue_channel)

        U_red, sing_values_red, V_red = np.linalg.svd(red_channel)
        U_green, sing_values_green, V_green = np.linalg.svd(green_channel)
        U_blue, sing_values_blue, V_blue = np.linalg.svd(blue_channel)

        sigma_red = np.zeros_like(red_channel)
        sigma_green = np.zeros_like(green_channel)
        sigma_blue = np.zeros_like(blue_channel)


        np.fill_diagonal(sigma_red, sing_values_red)
        np.fill_diagonal(sigma_green, sing_values_green)
        np.fill_diagonal(sigma_blue, sing_values_blue)
        top_k = stream.sidebar.slider('Регулируйте значение для более лучшей детализации', 1,len(sing_values_red),5)

        trunc_U_red = U_red[:, :top_k]
        trunc_sigma_red = sigma_red[:top_k, :top_k]
        trunc_V_red = V_red[:top_k, :]
        
        trunc_U_green = U_green[:, :top_k]
        trunc_sigma_green = sigma_green[:top_k, :top_k]
        trunc_V_green = V_green[:top_k, :]
        
        trunc_U_blue = U_blue[:, :top_k]
        trunc_sigma_blue = sigma_blue[:top_k, :top_k]
        trunc_V_blue = V_blue[:top_k, :]
        
        
        trunc_img_red = trunc_U_red @ trunc_sigma_red @ trunc_V_red
        trunc_img_green = trunc_U_green @ trunc_sigma_green @ trunc_V_green
        trunc_img_blue = trunc_U_blue @ trunc_sigma_blue @ trunc_V_blue
        
        
        trunc_img_red = (trunc_img_red - np.min(trunc_img_red)) / (np.max(trunc_img_red) - np.min(trunc_img_red))
        trunc_img_green = (trunc_img_green - np.min(trunc_img_green)) / (np.max(trunc_img_green) - np.min(trunc_img_green))
        trunc_img_blue = (trunc_img_blue - np.min(trunc_img_blue)) / (np.max(trunc_img_blue) - np.min(trunc_img_blue))
        
        
        reconstructed_image = np.dstack((trunc_img_red, trunc_img_green, trunc_img_blue))
        
        
        stream.image(reconstructed_image)
    except Exception as e:
        stream.write(f"Произошла ошибка при загрузке изображения: {str(e)}")

