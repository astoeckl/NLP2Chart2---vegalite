import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt, mpld3
from matplotlib.figure import Figure
from mpld3 import fig_to_html
import openai
import numpy as np
import random
import requests
import pickle
from streamlit.report_thread import get_report_ctx
from matplotlib.colors import to_hex
import os
from os import listdir
from dotenv import load_dotenv

import json
from nl4dv import NL4DV
import altair as alt

import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('all-corpora')

import spacy
nlp = spacy.load('en_core_web_sm') # instead of spacy.load('en')

load_dotenv()
token = os.environ.get("api-token")
openai.api_key = token

#markersdict = {'.': 'point', ',': 'pixel', 'o': 'circle', 'v': 'triangle_down', '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', '8': 'octagon', 's': 'square', 'p': 'pentagon', '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', 'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline', 'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright', 6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase', 10: 'caretupbase', 11: 'caretdownbase', 'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}

### Customize UI
st.set_page_config(layout="wide", page_title="NLP2Chart", page_icon=":)")
#st.set_page_config(layout="wide")

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {0}rem;
        padding-right: {5}rem;
        padding-left: {3}rem;
        padding-bottom: {0}rem;
    }} </style> """, unsafe_allow_html=True)


### Session ID vergeben

def get_session_id():
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id
    return session_id
st.session_state.id = str(get_session_id())

### Figure intialisieren

def init_widgets():
    if 'xaxis' in st.session_state:
        del st.session_state.xaxis
    if 'yaxis' in st.session_state:
        del st.session_state.yaxis
    if 'title' in st.session_state:
        del st.session_state.title
    if 'xlim_start' in st.session_state:
        del st.session_state.xlim_start
    if 'xlim_end' in st.session_state:
        del st.session_state.xlim_end
    if 'ylim_start' in st.session_state:
        del st.session_state.ylim_start
    if 'ylim_end' in st.session_state:
        del st.session_state.ylim_end
    return True

#fig = plt.gcf()
#contained_artists = fig.get_children()
#if len(contained_artists) <= 1:
#fig = plt.figure()
#ax = fig.add_subplot()
#ax.set_title("Figure Title")
#with open('fig'+ st.session_state.id +'.pickle', 'wb') as f: # should be 'wb' rather than 'w'
    #pickle.dump(fig, f)

### Layout Sidebar ###

#agree = st.sidebar.checkbox('Grid')
#st.session_state.grid = agree

st.sidebar.markdown('### Datasets ###')
# Data Selection
datafiles = ['No Dataset'] + [file for file in listdir(".") if file.endswith('.csv')]
option = st.sidebar.selectbox(
    'Which dataset do you want to use?',
     datafiles, key = 'dataset', on_change=init_widgets)

# File Upload
uploaded_file = st.sidebar.file_uploader("Or upload a file", type= ['csv'], key = 'fileupload', on_change=init_widgets)
if uploaded_file != None:
    try:
        dataframe = pd.read_csv(uploaded_file)
        dataframe.to_csv(uploaded_file.name)
        option = uploaded_file.name
    except:
        st.sidebar.write("Could not import")

if option in datafiles:
    ind = datafiles.index(option)
    if option != 'No Dataset':
        st.session_state.comand_load = "pd.read_csv(" + datafiles[ind]+"\')"
        if option == 'cars.csv':
            st.session_state.prompt_load = "https://raw.githubusercontent.com/astoeckl/NLP2Chart-Data/main/cars.csv"
        if option == 'califormia-housing.csv':
            st.session_state.prompt_load = "https://raw.githubusercontent.com/astoeckl/NLP2Chart-Data/main/california-housing.csv"
        if option == 'covid-data.csv':
            st.session_state.prompt_load = "https://raw.githubusercontent.com/astoeckl/NLP2Chart-Data/main/covid-data.csv"
        if option == 'gapminder-data.csv':
            st.session_state.prompt_load = "https://raw.githubusercontent.com/astoeckl/NLP2Chart-Data/main/gapminder-data.csv"
        df = pd.read_csv(datafiles[ind])
        data = pd.DataFrame(df.dtypes)
        data = data.astype(str)
        data.columns=['Type']
        data = data.Type.replace({'object': 'String','float64': 'Float','int64': 'Int'})
        st.sidebar.table(data)
    else:
        st.sidebar.write('No Dataset selected')
        st.session_state.comand_load = ''
        st.session_state.prompt_load = ''


### Layout Main ###

col1, col2 = st.columns([8,2])

def set_widgets():

    with open('vegafig'+ st.session_state.id +'.pickle', 'rb') as f: # should be 'wb' rather than 'w'
        vegafig = pickle.load(f)

    with col2:
        #print(vegafig)
        ### Widgets General
        try:
            figtitel = vegafig['title']['text']
        except:
            figtitel = ""
        st.text_input("Title", value = figtitel, key="title")


        axes_label = st.expander(label='Axes Label')
        with axes_label:
            st.text_input("Text for x-axis", value = "", key="xaxis")
            st.text_input("Text for y-axis", value = "", key="yaxis")

        axes_limits = st.expander(label='Axes Limits')
        with axes_limits:
            st.number_input("From for x-axis", step=1, value=0, key="xlim_start")
            st.number_input("To for x-axis", step=1, value=0, key="xlim_end")
            st.number_input("From for y-axis", step=1, value=0, key="ylim_start")
            st.number_input("To for y-axis", step=1, value=0, key="ylim_end")

        try:
            type = vegafig['mark']['type']
        except:
            type = None
        ### Widgets for Barplots
        if type=='bar':
            color_bars = st.expander(label='Bar Colors')
            with color_bars:
                color = "blue"
                color = to_hex(color)
                st.color_picker('Bar Color', color, key="barcolor")

        ### Widgets for Lines

        if type=='line':
            color_lines = st.expander(label='Line Colors')
            with color_lines:
                color = "blue"
                color = to_hex(color)
                st.color_picker('Line 1', color, key="linecolor")
            style_lines = st.expander(label='Line Style')
            with style_lines:
                st.selectbox("Line 1",  ('solid', 'dashed', 'dotted', 'dashdot'), index = 0, key="linestyle")

            width_lines = st.expander(label='Line Width')
            with width_lines:
                #for i in range(numlines):
                    #lwidth = plt.gca().get_lines()[i].get_linewidth()
                st.number_input("Line 1", min_value=0.0, max_value=10.0, step= 1.0, value=1.0, key="linewidth")


            #marker_lines = st.expander(label='Marker')
            #with marker_lines:
                #for i in range(numlines):
                    #lmarker = plt.gca().get_lines()[i].get_marker()
                    #if lmarker in markersdict.keys():
                        #ind = list(markersdict.keys()).index(lmarker)
                    #else:
                        #ind = 0
                    #st.selectbox("Line "+str(i+1),  markersdict.values(), index = ind, key="linemarker"+str(i))

            #legend_lines = st.expander(label='Legend')
            #with legend_lines:
                #if plt.gca().get_legend() != None:
                    #vis = plt.gca().get_legend().get_visible()
                #else:
                    #vis = False
                #st.checkbox('Visible', value = vis , key = 'visiblelegend')
                #for i in range(numlines):
                    #llegend = plt.gca().get_lines()[i].get_label()
                    #if llegend.startswith('_'):
                        #plt.gca().get_lines()[i].set_label('Line '+str(i))
                        #llegend = 'Line '+str(i+1)
                    #print(llegend)
                    #st.text_input("Label for line "+str(i+1), value = llegend, key="linelabel"+str(i))


        ### Widgets for Scatterplots #
        #if plt.gca().collections:
            #color_points = st.expander(label='Point Color Palettes')
            #numcolls = len(plt.gca().collections)
            #with color_points:
                #for i in range(numcolls):
                    #pmap = plt.getp(plt.gca().collections[i], 'cmap').name
                    #maplist = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
                    #maplist = plt.colormaps()
                    #if pmap in maplist:
                        #ind = maplist.index(pmap)
                    #else:
                        #ind = 0
                    #st.selectbox("Points "+str(i+1),  maplist, index = ind, key="pointcolor"+str(i))

        ### Widgets for Barplots
        #if plt.gca().containers:
            #bar_color = st.expander(label='Color of Bars')
            #num_bars = len(plt.gca().containers)
            #with bar_color:
                #for i in range(num_bars):
                    #color = plt.gca().containers[i].patches[0].properties()['facecolor']
                    #color = to_hex(color)
                    #st.color_picker('Bargroup '+str(i+1), color, key="barcolor"+str(i))


### Create Figure ###

#@st.cache
def create_figure():
    #data_url = os.path.join(".", "gapminder-data.csv")

    #data_url ="file://"+ ".\gapminder-data.csv"
    #print(data_url)
    data_url = st.session_state.prompt_load
    label_attribute = None
    dependency_parser_config = {"name": "spacy", "model": "en_core_web_sm", "parser": None}

    nl4dv_instance = NL4DV(verbose=False,
                       debug=True,
                       data_url=data_url,
                       label_attribute=label_attribute,
                       dependency_parser_config=dependency_parser_config
                       )
    nl4dv_response = nl4dv_instance.analyze_query(st.session_state.comand_input)

    #print(nl4dv_response['visList'][0]['vlSpec'])
    #if len(nl4dv_response['visList'])>0:
        #dict = nl4dv_response['visList'][0]['vlSpec']
    #else:
        #dict = {}

    #if dict['encoding']['x']['aggregate'] == None:
        #dict['encoding']['x'].pop('aggregate')
    #if dict['encoding']['y']['aggregate'] == None:
        #dict['encoding']['y'].pop('aggregate')

    #chart = alt.Chart.from_dict(dict)

    #chart = chart.properties(width=800, height=450)

    #chart = chart.properties(color='#000000')
    #chart = chart.mark_line().encode(color=alt.value("#FFAA00"))


    if len(nl4dv_response['visList'])>0:
        vegafig = nl4dv_response['visList'][0]['vlSpec']
        vegafig['width'] = 800
        vegafig['height'] = 450
    else:
        vegafig = {}

    #st.vega_lite_chart(data = df, spec=testvega)
    ### Statements from Data and GPT3
    #my_exec(st.session_state.comand_load)
    #print(st.session_state.comand_load)
    #if 'comand_output' in st.session_state:
        #my_exec(st.session_state.comand_output)


    # Statements from frontend
    ### Set Grid
    #my_exec(st.session_state.grid)
    #vegafig['config']['axis']['grid'] = st.session_state.grid
    #vegafig = chart.to_dict()
    #print(vegafig)
    ### Set Axes Labels

    if 'xaxis' in st.session_state:
        try:
            if st.session_state.xaxis != '':
                vegafig['encoding']['x']['title'] = st.session_state.xaxis
        except:
            pass
    if 'yaxis' in st.session_state:
        try:
            if st.session_state.yaxis != '':
                vegafig['encoding']['y']['title'] = st.session_state.yaxis
        except:
            pass

    ### Set Limits for the Axes
    if 'xlim_start' in st.session_state and 'xlim_end' in st.session_state:
        try:
            if not (st.session_state.xlim_start == 0 and st.session_state.xlim_end == 0):
                vegafig['encoding']['x']['scale'] ={}
                vegafig['encoding']['x']['scale']['domain'] = [st.session_state.xlim_start,st.session_state.xlim_end]
        except:
            pass
    if 'ylim_start' in st.session_state and 'ylim_end' in st.session_state:
        try:
            if not (st.session_state.ylim_start == 0 and st.session_state.ylim_end == 0):
                vegafig['encoding']['y']['scale'] ={}
                vegafig['encoding']['y']['scale']['domain'] = [st.session_state.ylim_start,st.session_state.ylim_end]
        except:
            pass

    #chart = alt.Chart.from_dict(vegafig)

    #numcolls = len(plt.gca().collections)
    #numlines = len(plt.gca().get_lines())
    #num_bars = len(plt.gca().containers)

    ### Set Colors
    if 'linecolor' in st.session_state:
        if st.session_state['linecolor'] != to_hex("blue"):
            try:
                vegafig['encoding']['color'] = {}
                vegafig['encoding']['color']['value'] = st.session_state['linecolor']
            except:
                pass


    ### Set Linestyle
    if 'linestyle' in st.session_state:
        if st.session_state['linestyle'] != "solid":
            try:
                #print(st.session_state['linestyle'])
                vegafig['encoding']['strokeDash'] = {}
                if st.session_state['linestyle'] == 'dotted':
                    vegafig['encoding']['strokeDash']['value'] = 1,1
                if st.session_state['linestyle'] == 'dashed':
                    vegafig['encoding']['strokeDash']['value'] = 4,4
                if st.session_state['linestyle'] == 'dashdot':
                    vegafig['encoding']['strokeDash']['value'] = 8,2,8
            except:
                pass
        else:
            try:
                vegafig['encoding']['strokeDash'] = {}
                vegafig['encoding']['strokeDash']['value'] = 1,0
            except:
                pass

    ### Set Linewidth
    if 'linewidth' in st.session_state:
        if st.session_state['linewidth'] != 1.0:
            try:
                vegafig['encoding']['strokeWidth'] = {}
                vegafig['encoding']['strokeWidth']['value'] = st.session_state['linewidth']
            except:
                pass

    ### Set Marker
    #reversed_markersdict = {value : key for (key, value) in markersdict.items()}
    #for i in range(numlines):
        #if 'linemarker'+str(i) in st.session_state:
            #mark = reversed_markersdict[st.session_state['linemarker'+str(i)]]
            #exec('plt.gca().get_lines()[i].set_marker(\''+ mark +'\')')
            #if plt.gca().get_legend() != None:
                #exec('plt.gca().get_legend().legendHandles[i].set_marker(\''+ mark +'\')')

    ### Set legend
    #for i in range(numlines):
        #if 'linelabel'+str(i) in st.session_state:
            #exec('plt.gca().get_lines()[i].set_label(\''+ st.session_state['linelabel'+str(i)] +'\')')
            #plt.legend()

    #if 'visiblelegend' in st.session_state:
        #if st.session_state.visiblelegend:
            #if plt.gca().get_legend() != None:
                #plt.gca().get_legend().set_visible(True)
        #else:
            #if plt.gca().get_legend() != None:
                #plt.gca().get_legend().set_visible(False)

############## Scatterplots #####

    ### Set Colormap
    #for i in range(numcolls):
        #if 'pointcolor'+str(i) in st.session_state:
            #plt.setp(plt.gca().collections[i], cmap=plt.get_cmap(st.session_state['pointcolor'+str(i)]))
        #plt.getp(plt.gca().collections[i], 'cmap').name

############## Barcharts ##########

    ### Set Colors of Bargroups
    if 'barcolor' in st.session_state:
        if st.session_state['barcolor'] != to_hex("blue"):
            try:
                vegafig['encoding']['color'] = {}
                vegafig['encoding']['color']['value'] = st.session_state['barcolor']
            except:
                pass

######## Set Title
    if 'title' in st.session_state:
        #print(chart.properties())
        #chart = chart.properties(title = st.session_state.title)
        vegafig['title'] = {
            "text": st.session_state.title
            }

    with open('vegafig'+ st.session_state.id +'.pickle', 'wb') as f: # should be 'wb' rather than 'w'
        pickle.dump(vegafig, f)

    return vegafig


with col1:
    #st.write(st.session_state)
    st.header('Create Charts with Commands in Natural Language')
    demo_video = st.expander(label='Tutorial Video')
    with demo_video:
        st.video(data="https://youtu.be/UiCSczhslAs")
    st.text_area("Advise the system", key="comand_input",
        help="Examples: \n Plot a sinus function from -4 pi to 4 pi; \n Make an array of 400 random numbers and plot a horizontal histogram; \n plot sum of total_cases grouped by location as bar chart (COVID19 Data)")
    fig = create_figure()
    #st.pyplot(fig=fig)

    if fig != {}:
        st.vega_lite_chart(spec=fig, use_container_width=True)

set_widgets()

### Export Figures

#st.sidebar.markdown('### Export ###')

#with open('vegafig'+ st.session_state.id +'.pickle', 'rb') as f:
    #fig = pickle.load(f)
#chart = alt.Chart.from_dict(fig)
#chart.save('figure_export.png')
#chart.save('figure_export.html')
#fig.savefig('figure_export.png', dpi=fig.dpi)
#mpld3.save_html(fig,'figure_export.html')
#with open('figure_export.png', 'rb') as f:
   #st.sidebar.download_button('Download PNG', f, file_name='figure_export.png')
#with open('figure_export.html', 'rb') as f:
   #st.sidebar.download_button('Download HTML', f, file_name='figure_export.html')
