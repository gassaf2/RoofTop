import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image,ImageDraw
from io import StringIO
import streamlit as st
#import folium
#from streamlit_folium import folium_static
#import cv2
import tensorflow as tf
from keras.models import load_model
       

# st.header('Roof Top Detection', divider='blue',help='This is an application where user can drop an image and the system will detect the roof top  using deep learning model')
st.header('Deep Learning application')

st.caption(' Introduction: The usage of this application is free of cost')

# Definition of global variables - GA

with st.sidebar:
    
    model1="model1.h5"
    model2="model2.h5"
    model3="model3.h5"
    
    dict_model={
        model1:"Model trained for 30 epochs<br>Accuracy: 0.7<br>Loss: 0.2<br>",
        model2:"Accuracy: 0.8<br>Loss: 0.3",
        model3:"Accuracy: 0.95<br>Loss:0.1"
    }
    option = st.selectbox(
    '**Select a model**',
    (model1, model2,model3))

    
    st.markdown("**Parameters of selected model:**")
    st.write(dict_model[option],unsafe_allow_html=True)
  
var2=0

transparent_image0 = Image.new("RGBA",(250, 250), (0,0, 0, 0))
transparent_image1 = Image.new("RGBA",(250, 250), (0,0, 0, 0))
# Definition of python definition
def on_button_predict():
    st.write("Button clicked!")

img_size=(250,250)

def test(arg1):
    return arg1

def plot_predicted(img_pred):
    st.write("image is predicted")
    img_pred=np.squeeze(img_pred, axis=0)
    for index in range(img_pred.shape[0]):
        img=img_pred[index,:,:]
        #transparent_image=f"result_streamlit/pred0.png"    
        col2.image(img,caption=f"Mask {index} - pred{index}")

# def load_images(file_list):
#     data = {'File_Name': []}
    
#     for file in file_list:
#         file_name=file.name
#         data['File_Name'].append(file_name)
#     #create a dataframe   
#     df = pd.DataFrame(data)
#         # Display DataFrame using st.write
#     st.write("Using st.write:")
#     st.write(df)
#     return df  

def load_images(file_list):
    data = {'file': []}
    
    for f in file_list:
        
        data['file'].append(f.getvalue())
    #create a dataframe   
    df = pd.DataFrame(data)
        # Display DataFrame using st.write
    st.write("Using st.write:")
    st.write(df)
    return df  

img_size=(250,250)

def preprocessing(img_byte):
    #solar_img = tf.io.read_file(solar_path) 
    
    solar_img=tf.io.decode_image(img_byte, channels=3, dtype=tf.float32,expand_animations=False)
    #solar_img = tf.image.decode_image(solar_img, channels=3, dtype=tf.float32,expand_animations=False)
    solar_img = tf.image.resize(solar_img,img_size)
    solar_img = tf.cast(solar_img, tf.float32) / 255.0
    #solar_img=tf.reshape(solar_img, (250,250,3))
    return solar_img


def create_dataset(df, train = False):
    if not train:
        ds = tf.data.Dataset.from_tensor_slices((df["file"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((df["file"].values))
        ds = ds.map(preprocessing, tf.data.AUTOTUNE)
    return ds 


def pred(model, dataset, threshold, num=1):
    # store the predicted values in a temporary dataset
    img_pred=[]
    for img in dataset.take(num):
        img_pred.append(model.predict(img))
        
    # Mask the predicted output
    temp=img_pred
    temp = np.array(temp)
    temp[temp >= threshold] = 1
    temp[temp < threshold] = 0
    return temp

def convert_numpy_to_img(output_directory,array_img_list):
    # Save each image in the list
    print(array_img_list.shape)
    for i in range(array_img_list.shape[0]):
        image_array=array_img_list[i]
        print(image_array.shape)
        
         # Convert the float array to integers in the valid range
        int_array = (image_array * 255).astype(np.uint8)

    # Reshape the array to remove singleton dimensions
        image_array = np.squeeze(int_array)
        
        
        image_pil=Image.fromarray(image_array)
        
        image_filename=os.path.join(output_directory,f"pred{i}.png")
        
        image_pil.save(image_filename)

#core code

list_images_path=[]
uploaded_files=st.file_uploader("Choose a image file" , accept_multiple_files=True,type=("jpg","png"))
#if uploaded_files is not None:
nbr_images=len(uploaded_files)


st.markdown(f"Nbr of Images uploaded is: {nbr_images}")


if len(uploaded_files) >0 :    
    #print("Uploaded File:", type(uploaded_files))
    num_rows=nbr_images
    num_cols=2
    if uploaded_files is not None and len(uploaded_files)>=0:
        
       # df=load_data(uploaded_files) # load the data in dataframe
         # Display DataFrame
       # st.write("Displaying DataFrame:")
       # st.write(df)
       
        col1,col2=st.columns(2)
        col1.header("Original Image")
        col2.header("Predicted Image")
       
        for row in range(num_rows):
            
            #create a rows with num_cols columns
            #col1,col2=st.columns(2)
            
            image=Image.open(uploaded_files[row])
            col1.image(image,caption=f"Image {row} - {uploaded_files[row].name}")
        button=st.button("Predict",type="primary", on_click=on_button_predict)
    
        if button:
            test_df=load_images(uploaded_files)
            
            st.write(test_df)
            test_dataset = create_dataset(test_df)
            test_dataset=test_dataset.batch(num_rows)
            st.write(test_dataset)
            
            model = load_model('model_streamlit.h5') 
            
            img_pred=pred(model, test_dataset, 0.5, num_rows) 
            
            st.write("Shape of img_prod",img_pred.shape)
           # convert_numpy_to_img("pred_imgs",img_pred)
            
            plot_predicted(img_pred)
           
def test(arg1):
    return arg1





st.image("ai-article-pic8.jpeg",use_column_width ="auto")
     
 
st.markdown(var2, unsafe_allow_html=False, help=None,)
# Coordinates for a location (e.g., San Francisco, CA)



    
    
    
    
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

# # Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )

# tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])
# uploaded_file = st.file_uploader("Choose an image...", type="jpg, jpeg, tiff")
# with tab1:
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

# with tab2:
#    st.header("A dog")
#    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

# with tab3:
#    st.header("An owl")
#    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
