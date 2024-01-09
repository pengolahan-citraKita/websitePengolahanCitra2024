# -*- coding: utf-8 -*-
"""
Created on Tue Jan 2 2024

@author: febri, jaoza, annisa, armeisa
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Landing Page','Histogram','Transformasi Geometri', 'Morfologi Citra', 'Thresholding', 'Filtering', 'Video', 'Face Detection', 'Feature Detection', 'Object Detection')
    )
    # pilihan menu

    if selected_box == 'Landing Page':
        landingpage() 
    if selected_box == 'Histogram':
        histogram() 
    if selected_box == 'Transformasi Geometri':
        transGeo()
    if selected_box == 'Morfologi Citra':
        morfCitra()
    if selected_box == 'Thresholding':
        photo()
    if selected_box == 'Filtering':
        filter()
    if selected_box == 'Video':
        video()
    if selected_box == 'Face Detection':
        face_detection()
    if selected_box == 'Feature Detection':
        feature_detection()
    if selected_box == 'Object Detection':
        object_detection() 
 

def landingpage():
    
    st.title('Sistem Pengolahan Citra Untuk Pemrosesan Gambar')
    
    st.subheader('sebuah sistem sederhana yang dirancang untuk mengolah citra dengan tujuan melakukan pemrosesan gambar secara efisien dan efektif. Sistem ini mencakup beberapa teknik pengolahan citra yang dapat diterapkan untuk meningkatkan kualitas, mengoptimalkan informasi, dan menghasilkan gambar yang lebih baik.')
    
    st.image('image-processing.jpg',use_column_width=True)


def load_image(filename):
    image = cv2.imread(filename)
    return image

def histogram():
    st.header("Histogram")
    st.subheader("Histogram citra merupakan diagram yang menggambarkan frekuensi setiap nilai intensitas yang muncul di seluruh piksel citra.")
    if st.button('Lihat Citra Asli'):
        
        original = Image.open('lena.jpg')
        st.image(original, use_column_width=True)
    
    image = cv2.imread('lena.jpg',0)
    
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    st.bar_chart(histr)

def transGeo():
    st.header("Transformasi Geometri")
    st.write("Transformasi geometri terdiri dari beberapa jenis salah satunya ada Translasi dan Rotasi. Dapat dilihat pada contoh dibawah perbedaan diantara keduanya.")
    st.text("\n")

    #Translasi
    st.subheader("Translasi")
    if st.button('Citra Asli Translasi'):
            
        original = Image.open('groot.jpg')
        st.image(original, use_column_width=True)
    
    image = cv2.imread('groot.jpg',0)
    st.write("Berikut hasil citra yang telah dilakukan operasi translasi")

    imggambar = cv2.imread("groot.jpg", -1)

    height, width = imggambar.shape[:2]

    height, width = imggambar.shape[:2]

    quarter_height, quarter_width = -height/2, width/2
    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
    img_translation = cv2.warpAffine(imggambar, T, (width, height))

    # Menampilkan hasil translasi di Streamlit
    st.image(img_translation, caption="Hasil Translasi", use_column_width=True)

    st.text("\n")

    #Rotasi
    st.subheader("Rotasi")
    if st.button('Citra Asli Rotasi'):
            
        original = Image.open('olaf.jpg')
        st.image(original, use_column_width=True)
    
    imggambar= cv2.imread("olaf.jpg",-1)
    st.write("Berikut hasil citra yang telah dilakukan operasi rotasi")

    rows,cols = imggambar.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    imgrotasi = cv2.warpAffine(imggambar,M,(cols,rows))
    tampil_hor=np.concatenate((imggambar,imgrotasi),axis=1)

    # Menampilkan hasil translasi di Streamlit
    st.image(tampil_hor, caption="Hasil Rotasi", use_column_width=True)

def morfCitra():
    st.header("Morfologi Citra")
    st.write("Morfologi Citra merupakan salah satu teknik dalam pengolahan citra yang berkaitan dengan struktur, bentuk, dan hubungan spasial antar objek dalam citra. Terdapat beberapa jenis operasi morfologi citra salah satunya adalah erosi dan dilasi.")
    st.text("\n")

    #Erosi
    st.subheader("Erosi")
    if st.button('Citra Asli Erosi'):
            
        original = Image.open('bitcoin.jpg')
        st.image(original, use_column_width=True)
    
    imggambar= cv2.imread("bitcoin.jpg",-1)
    st.write("Berikut hasil citra yang telah dilakukan operasi erosi")

    imggambar = cv2.imread("bitcoin.jpg",0)
    kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((1,1),np.uint8)

    imgCanny = cv2.Canny(imggambar,10,150)
    imgdilation5 = cv2.dilate(imgCanny,kernel,iterations=1)
    imgErode = cv2.erode(imgdilation5,kernel2,iterations=1)

    tampil_hor=np.concatenate((imgCanny,imgErode),axis=1)
    
    # Menampilkan hasil erosi di Streamlit
    st.image(tampil_hor, caption="Hasil Erosi", use_column_width=True)

    st.text("\n")

    #Dilasi
    st.subheader("Dilasi")
    if st.button('Citra Asli Dilasi'):
            
        original = Image.open('daun.jpg')
        st.image(original, use_column_width=True)
    
    imggambar= cv2.imread("daun.jpg",-1)
    st.write("Berikut hasil citra yang telah dilakukan operasi dilasi")

    imggambar = cv2.imread("daun.jpg",0)
    kernel = np.ones((1,1),np.uint8)
    kernel2 = np.ones((15,15),np.uint8)

    imgCanny = cv2.Canny(imggambar,10,150)
    imgdilation = cv2.dilate(imgCanny,kernel,iterations=1)
    imgdilation2 = cv2.dilate(imgCanny,kernel2,iterations=1)
    imgdilation3= ~imgdilation2

    tampil_hor=np.concatenate((imggambar,imgdilation2),axis=0)
    
    # Menampilkan hasil erosi di Streamlit
    st.image(tampil_hor, caption="Hasil Dilasi", use_column_width=True)

    st.text("\n")

def photo():
    # nama titlenya
    st.header("Thresholding")
    # nama deskripsinya
    st.write("Thresholding merupakan salah satu segmentasi citra yang mana prosesnya didasarkan pada tingkat kecerahan atau gelap terangnya")

    # untuk melakukan upload file dalam bentuk jpg, jpeg, dan png
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    # apbila file yang diup tidak kosong (ada isinya)
    if uploaded_file is not None:
        # maka file image yang diup akan dibaca
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), -1)

        # kemudian akan dicek apakah file image yang diup sudah grayscale atau belum
        # jika sudah grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            # maka tidak perlu dilakukan konversi
            image = img 
        # apabila belum grayscale
        else:
            # maka akan dilakukan perubahan atau konversi ke grayscale
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # setelah itu dilakukanlah thresholding dari file yang telah diup
        x = st.slider('Change Threshold value', min_value=50, max_value=255)
        ret, thresh1 = cv2.threshold(image, x, 255, cv2.THRESH_BINARY)
        # kemudian ditampilkan hasil dari thresholdingnya
        st.image(thresh1, use_column_width=True, clamp=True)

# fungsi dari filter mean harmonik beserta argumennya img dan kernel_size yang bernilai 3
def apply_harmonic_mean_filter(img, kernel_size=3):
    # melakukan calculasi padding yang dibutuhkan di tepi gambar untuk mengakomodasikan kernel filter
    pad = kernel_size // 2
    # membuat array kosong
    filtered_img = np.zeros_like(img)

    # melakukan perulangan untuk setiap piksel di dalam image, tidak termasuk pixel batas karena padiing
    for i in range(pad, img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):
            # melakukan ekstrasi pada image berdasarkan ukuran kernal
            patch = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # melakukan perhitungan harmonic mean dari nilai pixel dan menetapkannya ke piksel yang sesuai pada gambar yang difilter
            filtered_img[i, j] = len(patch.flatten()) / np.sum(1 / patch.flatten())
    # mengonversi gambar yang difilter menjadi bilangan bulat 8-bit yang tidak ditandatangani (format gambar umum) dan mengembalikannya.
    return filtered_img.astype(np.uint8)

# fungsi dari filter mean harmonik beserta argumennya img dan kernel_size yang bernilai 5
def apply_harmonic_mean_filter5(img, kernel_size=5):
    # melakukan calculasi padding yang dibutuhkan di tepi gambar untuk mengakomodasikan kernel filter
    pad = kernel_size // 2
    # membuat array kosong
    filtered_img = np.zeros_like(img)

    # melakukan perulangan untuk setiap piksel di dalam image, tidak termasuk pixel batas karena padiing
    for i in range(pad, img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):
            # melakukan ekstrasi pada image berdasarkan ukuran kernal
            patch = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # melakukan perhitungan harmonic mean dari nilai pixel dan menetapkannya ke piksel yang sesuai pada gambar yang difilter
            filtered_img[i, j] = len(patch.flatten()) / np.sum(1 / patch.flatten())

    # mengonversi gambar yang difilter menjadi bilangan bulat 8-bit yang tidak ditandatangani (format gambar umum) dan mengembalikannya.
    return filtered_img.astype(np.uint8)

# fungsi dari filter mean harmonik beserta argumennya img dan kernel_size yang bernilai 9
def apply_harmonic_mean_filter9(img, kernel_size=9):
    # melakukan calculasi padding yang dibutuhkan di tepi gambar untuk mengakomodasikan kernel filter
    pad = kernel_size // 2
    # membuat array kosong
    filtered_img = np.zeros_like(img)

    # melakukan perulangan untuk setiap piksel di dalam image, tidak termasuk pixel batas karena padiing
    for i in range(pad, img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):
            # melakukan ekstrasi pada image berdasarkan ukuran kernal
            patch = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # melakukan perhitungan harmonic mean dari nilai pixel dan menetapkannya ke piksel yang sesuai pada gambar yang difilter
            filtered_img[i, j] = len(patch.flatten()) / np.sum(1 / patch.flatten())

    # mengonversi gambar yang difilter menjadi bilangan bulat 8-bit yang tidak ditandatangani (format gambar umum) dan mengembalikannya.
    return filtered_img.astype(np.uint8)

def filter():
    # nama titlenya
    st.header("Filtering Mean Harmonik")
    # nama deskripsinya
    st.write("Filtering Mean Harmonik merupakan salah satu metode yang"+
             "memiliki teknik cara bekerja dengan mengubah atau menggantikan"+
             "intensitas pixel dengan rata rata pixel dari pixel tetangganya.")
    # untuk melakukan upload file dalam bentuk jpg, jpeg, dan png
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    # jika file tidak kosong (ada isinya)
    if uploaded_file is not None:
        # maka file image akan dibaca
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # kemudian akan dilakukan filtering sesuai dengan fungsi yang telah dibuat
        # akan dilakukan filtering dengan kernel 3
        filtered_img = apply_harmonic_mean_filter(img)
        # akan dilakukan filtering dengan kernel 5
        filtered_img5 = apply_harmonic_mean_filter5(img)
        # akan dilakukan filtering dengan kernel 9
        filtered_img9 = apply_harmonic_mean_filter9(img)
        
        # ini merupakan caption yang berada di bawah gambar
        captions = ["Original Image", "Filtering Mean Harmonik (Kernel 3)", "Filtering Mean Harmonik (Kernel 5)", "Filtering Mean Harmonik (Kernel 9)"]
        # akan menampilkan gambar orignal beserta captionnya
        st.image(img, width=250, caption=captions[0])
        st.write("")
        # akan menampilkan gambar yang telah difiltering dengan kernel 3 beserta captionnya
        st.image(filtered_img, width=250, caption=captions[1])
        st.write("")
        # akan menampilkan gambar yang telah difiltering dengan kernel 5 beserta captionnya
        st.image(filtered_img5, width=250, caption=captions[2])
        st.write("")
        # akan menampilkan gambar yang telah difiltering dengan kernel 9 beserta captionnya
        st.image(filtered_img9, width=250, caption=captions[3])
        
            
    
def video():
    uploaded_file = st.file_uploader("Choose a video file to play")
    if uploaded_file is not None:
         bytes_data = uploaded_file.read()
 
         st.video(bytes_data)
         
    video_file = open('typing.mp4', 'rb')
         
 
    video_bytes = video_file.read()
    st.video(video_bytes)
 

def face_detection():
    
    st.header("Face Detection using haarcascade")
    
    if st.button('See Original Image'):
        
        original = Image.open('friends.jpeg')
        st.image(original, use_column_width=True)
    
    
    image2 = cv2.imread("friends.jpeg")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image2)
    print(f"{len(faces)} faces detected in the image.")
    for x, y, width, height in faces:
        cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    
    cv2.imwrite("faces.jpg", image2)
    
    st.image(image2, use_column_width=True,clamp = True)
 

def feature_detection():
    st.subheader('Feature Detection in images')
    st.write("SIFT")
    image = load_image("tom1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()    
    keypoints = sift.detect(gray, None)
     
    st.write("Number of keypoints Detected: ",len(keypoints))
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image, use_column_width=True,clamp = True)
    
    
    st.write("FAST")
    image_fast = load_image("tom1.jpg")
    gray = cv2.cvtColor(image_fast, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    st.write("Number of keypoints Detected: ",len(keypoints))
    image_  = cv2.drawKeypoints(image_fast, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image_, use_column_width=True,clamp = True)

    
    
def object_detection():
    
    st.header('Object Detection')
    st.subheader("Object Detection is done using different haarcascade files.")
    img = load_image("clock.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    clock = cv2.CascadeClassifier('haarcascade_wallclock.xml')  
    found = clock.detectMultiScale(img_gray,  
                                   minSize =(20, 20)) 
    amount_found = len(found)
    st.text("Detecting a clock from an image")
    if amount_found != 0:  
        for (x, y, width, height) in found:
     
            cv2.rectangle(img_rgb, (x, y),  
                          (x + height, y + width),  
                          (0, 255, 0), 5) 
    st.image(img_rgb, use_column_width=True,clamp = True)
    
    
    st.text("Detecting eyes from an image")
    
    image = load_image("eyes.jpg")
    img_gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    img_rgb_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
    eye = cv2.CascadeClassifier('haarcascade_eye.xml')  
    found = eye.detectMultiScale(img_gray_,  
                                       minSize =(20, 20)) 
    amount_found_ = len(found)
        
    if amount_found_ != 0:  
        for (x, y, width, height) in found:
         
            cv2.rectangle(img_rgb_, (x, y),  
                              (x + height, y + width),  
                              (0, 255, 0), 5) 
        st.image(img_rgb_, use_column_width=True,clamp = True)
    
    
    
    
if __name__ == "__main__":
    main()
