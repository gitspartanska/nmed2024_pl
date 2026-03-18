import streamlit as st
import json
import random
import pandas as pd
import pickle
import json

# set page configuration to wide mode
st.set_page_config(layout="wide")
st.badge("agpl-3.0 license", color="blue")
# section 1
st.markdown("#### O modelu")
st.markdown("Diagnostyka różnicowa demencji stanowi jedno z głównych wyzwań neurologii, głównie ze względu na nakładanie się objawów wynikających z różnych etiologii. Jednocześnie ma ona kluczowe znaczenie dla opracowania wczesnych, spersonalizowanych strategii leczenia. W niniejszym artykule przedstawiamy model sztucznej inteligencji, który integruje szeroki zakres danych (w tym dane demograficzne, historię medyczną pacjenta i jego rodziny, informacje o stosowaniu leków, wyniki ocen neuropsychologicznych i funkcjonalnych oraz multimodalne dane neuroobrazowe) w celu identyfikacji czynników przyczyniających się do rozwoju demencji u poszczególnych osób.")
st.markdown("Linki:\n* Publikacja: [https://www.nature.com/articles/s41591-024-03118-z](https://www.nature.com/articles/s41591-024-03118-z)\n* GitHub: [https://github.com/vkola-lab/nmed2024](https://github.com/vkola-lab/nmed2024)\n* Twórcy modelu: [https://vkola-lab.github.io/](https://vkola-lab.github.io/)")

# section 2
st.markdown("#### Minimodel")
st.markdown("Model został zaadaptowany z Hugging Face Space. Wykorzystuje ponad 300 parametrów na temat stanu pacjenta do oszacowania prawdopodobieństwa obecności u pacjenta wybranych schorzeń związanych z mózgiem.")
st.markdown("Linki:\n* HuggingFace: [https://huggingface.co/spaces/vkola-lab/nmed2024](https://huggingface.co/spaces/vkola-lab/nmed2024)\n* GitHub: [https://github.com/gitspartanska/nmed2024_pl](https://github.com/gitspartanska/nmed2024_pl)\n")
st.markdown("Aby skorzystać z minimodelu:\n* Wprowadź dane wejściowe w poniższym formularzu. Nie wszystkie pola muszą być wypełnione.\n* Kliknij przycisk \"**LOSOWY PRZYKŁAD**\", aby uzupełnić formularz losowym zestawem parametrów.\n* Użyj przycisku \"**PREDYKCJA**\", aby przesłać dane z formularza do modelu i otrzymać w tabeli przewidywane wyniki.\n* Aby zapisać tabelę z wynikami w formacie .csv, .tsv lub .txt, kliknij odpowiedni przycisk na dole strony.")

# load model
@st.cache_resource
def load_model():
    import adrd
    # try:   
    # except:
    #     # ckpt_path = '../adrd_tool_copied_from_sahana/dev/ckpt/ckpt_swinunetr_stripped_MNI.pt'
    #     ckpt_path = '/data_1/skowshik/ckpts_backbone_swinunet/ckpt_swinunetr_stripped_MNI.pt'
    #     model = adrd.model.ADRDModel.from_ckpt(ckpt_path, device='cpu')
    ckpt_path = 'ckpt_swinunetr_stripped_MNI.pt'
    model = adrd.model.ADRDModel.from_ckpt(ckpt_path, device='cpu')
    return model

model = load_model()

def predict_proba(data_dict):
    pred_dict = model.predict_proba([data_dict])[1][0]
    return pred_dict

# load meta data csv
file_path = './data/input_meta_info.csv'
input_meta_info = pd.read_csv(file_path)

# load NACC testing data
from data.dataset_csv import CSVDataset
dat_tst = CSVDataset(
    dat_file = "./data/test_public.csv", 
    cnf_file = "./data/input_meta_info.csv"
)

def get_random_example():
    idx = random.randint(0, len(dat_tst) - 1)
    random_case = dat_tst[idx][0]
    return random_case

# Get random example features if the button is clicked
if 'random_example' not in st.session_state:
    st.session_state.random_example = None

st.markdown('---')
cols = st.columns(3)
with cols[1]:
    random_example_button = st.button("LOSOWY PRZYKŁAD", use_container_width=True)
if random_example_button:
    st.session_state.random_example = get_random_example()
    st.rerun()

random_example = st.session_state.random_example

def create_input(df, i):
    row = df.iloc[i]
    name = row['Name']
    description = row['Description']

    # dirty work, inspect keys and values
    values = row['Values']
    values = values.replace('\'', '\"')
    values = values.replace('\"0\": nan, ', '')
    values = json.loads(values)

    for k, v in list(values.items()):
        if v == 'Nieznany':
            values.pop(k)
        elif k in ('9', '99', '999'):
            values.pop(k)

        # get default value from random example if available
    default_value = random_example[name] if random_example and name in random_example else None
    if type(default_value) is float:
        default_value = int(default_value)

    # Determine the type of widget based on values
    if 'zakres' in values:
        if ' - ' in values['zakres']:
            min_value, max_value = map(float, values['zakres'].split(' - '))
            min_value, max_value = int(min_value), int(max_value)

            if default_value is not None:
                if default_value > max_value or default_value < min_value:
                    default_value = None
                
            st.number_input(description, key=name, min_value=min_value, max_value=max_value, value=default_value, placeholder=values['zakres'])
        else:
            min_value = int(values['zakres'].replace('>= ', ''))
            if default_value is not None:
                if default_value < min_value or default_value == 8888:
                    default_value = None

            st.number_input(description, key=name, min_value=min_value, value=default_value, placeholder=values['zakres'])
    else:
        values = {int(k): v for k, v in values.items()}
        if default_value in values:
            default_index = list(values.keys()).index(default_value)
        else:
            default_index = None

        st.selectbox(
            description, 
            options = values.keys(), 
            key = name, 
            index = default_index,
            format_func=lambda x: values[x],
            placeholder="Wybierz opcję"
        )

# create form
with st.form("dynamic_form"):
    sections = input_meta_info['Section'].unique()
    for section in sections:
        with st.container():
            st.markdown(f"##### {section}")
            sub_df = input_meta_info[input_meta_info['Section'] == section]

            cols = st.columns(3)
#            for c in cols:
#                print(c)
            with cols[0]:
                for i in range(0, len(sub_df), 3):
                    create_input(sub_df, i)
            with cols[1]:
                for i in range(1, len(sub_df), 3):
                    create_input(sub_df, i)
            with cols[2]:
                for i in range(2, len(sub_df), 3):
                    create_input(sub_df, i)    

        # seperate line
        st.markdown("---")
    cols = st.columns(3)
    with cols[1]:
        predict_button = st.form_submit_button("PREDYKCJA", use_container_width=True, type='primary')

# load mapping
with open('./data/nacc_variable_mappings.pkl', 'rb') as file:
    nacc_mapping = pickle.load(file)

def convert_dictionary(original_dict, mappings,is_SI=False):
    transformed_dict = {}
    
    for key, value in original_dict.items():
        if key in mappings:
            if is_SI:
                if key=="HEIGHT" and not value is None:
                    # cm -> inches 
                    value = value*0.3937
                elif key == "WEIGHT" and not value is None:
                    # Kg -> Ibs
                    value = value*2.2

            new_key, transform_map = mappings[key]
            
            # If the value needs to be transformed
            if value in transform_map:
                transformed_value = transform_map[value]
            else:
                transformed_value = value  # Keep the original value if no transformation is needed
            
            transformed_dict[new_key] = transformed_value
    
    return transformed_dict

if predict_button:
    # get form input
    names = input_meta_info['Name'].tolist()
    data_dict = {}
    for name in names:
        data_dict[name] = st.session_state[name]
    
    # convert + from SI
    data_dict = convert_dictionary(data_dict, nacc_mapping,True)
    pred_dict = predict_proba(data_dict)

    # change key name and value representations
    key_mappings = {
        'NC': 'Normalne poznanie',
        'MCI': 'Łagodne zaburzenia procesów poznawczych',
        'DE': 'Demencja',
        'AD': 'Choroba Alzheimera',
        'LBD': 'Otępienie z ciałami Lewy\'ego i choroba Parkinsona',
        'VD': 'Uraz naczyniowy mózgu lub otępienie naczyniopochodne obejmujące udarowe',
        'PRD': 'Choroby prionowe, w tym choroba Creutzfeldta-Jakoba',
        'FTD': 'Otępienie czołowo-skroniowe',
        'NPH': 'Wodogłowie normotensyjne',
        'SEF': 'Czynniki systemowe i zewnętrzne',
        'PSY': 'Choroby psychatryczne',
        'TBI': 'Urazowe uszkodzenie mózgu',
        'ODE': 'Inne przyczyny, obejmujące nowotwory, zanik wieloukładowy, drżenie samoistne, chorobę Huntingtona, zespół Downa i napady padaczkowy'
    }
    val = list(pred_dict.values())
    pred_dict = {key_mappings[k]: f"{v * 100:.2f}%" for k, v in pred_dict.items()}

    df = pd.DataFrame(list(pred_dict.items()), columns=['Etykieta', 'Przewidziane prawdopodobieństwo'])
    st.table(df)
    df2 = pd.DataFrame(list({a:round(b*100,2) for a,b in enumerate(val)}.items()),
        columns=['Etykieta', 'Przewidziane prawdopodobieństwo'])

    col = ["reszta" for i in val]
    col[df2.idxmax()['Przewidziane prawdopodobieństwo']]="największa wartość"
    df2["color"]=col
    st.bar_chart(df2, x = 'Etykieta', y = 'Przewidziane prawdopodobieństwo',
        y_label= "% prawdopodobieństwo",color = "color")

    cols = st.columns(3)

    with cols[0]:
        st.download_button(
            label="Pobierz wyniki jako .csv",
            data=df.to_csv(sep=',', encoding="utf-8"),
            file_name="wyniki.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
    with cols[1]:
        st.download_button(
            label="Pobierz wyniki jako .tsv",
            data=df.to_csv(sep='\t', encoding="utf-8"),
            file_name="wyniki.tsv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )
    with cols[2]:
        txt_data=f"   Przewidziane prawdopodobieństwo Etykieta\n"
        i =0
        for k, v in pred_dict.items():
            txt_data+=f"{i}{' ' if i<10  else ''}{' '*(32-len(v))}{v} {k}\n"
            i+=1
        st.download_button(
            label="Pobierz wyniki jako .txt",
            data=txt_data,
            file_name="wyniki.txt",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )

st.markdown("#### Słownik pojęć i skrótów")
st.markdown("\n* **Wskaźnik Ischemiczny Hachinskiego (HIS)** - Wskaźnik Hachinskiego (HIS) to punktowy system oceny objawów i czynników ryzyka, który pozwala klinicznie odróżnić demencję naczyniową od choroby Alzheimera, ułatwiając wczesną diagnozę i planowanie leczenia.")
st.markdown("\n* **NPI‑Q (Neuropsychiatric Inventory – Questionnaire)** to krótka wersja kwestionariusza Neuropsychiatric Inventory (NPI), służąca do oceny objawów neuropsychiatrycznych u osób z zaburzeniami poznawczymi, najczęściej w przebiegu demencji, w tym choroby Alzheimera.")
st.markdown("\n* **Osoba współtowarzysząca w NPI-Q** - osoba współtowarzysząca odpowiada na pytania w NPI‑Q w imieniu badanego.")
st.markdown("\n* **Skala Depresji Geriatrycznej (Geriatric Depression Scale, GDS)** to standaryzowany kwestionariusz przesiewowy używany do oceny nasilenia objawów depresji u osób starszych.")
st.markdown("\n* **Kwestionariusz Aktywności Funkcjonalnych (Functional Activities Questionnaire, FAQ)** to narzędzie oceniające zdolność osób dorosłych, szczególnie starszych, do wykonywania codziennych czynności instrumentalnych. Jest szeroko stosowany w diagnostyce zaburzeń poznawczych i demencji.")
st.markdown("\n* **MMSE (Mini-Mental State Examination)** to prosty test pozwalający ocenić ogólny stan poznawczy pacjenta, wykrywając problemy z pamięcią, orientacją, uwagą i językiem, stosowany m.in. w diagnostyce demencji.")
st.markdown("\n* **Logical Memory IA** to badanie pamięci zaraz po przeczytaniu historii.")
st.markdown("\n* **Logical Memory IIA** to badanie pamięci po pewnym czasie od przeczytania historii.")
st.markdown("\n* **WAIS‑R Digit Symbol** to test mierzący szybkość przetwarzania informacji, uwagę i sprawność psychomotoryczną, polegający na szybkim przypisywaniu symboli do cyfr zgodnie z legendą.")
st.markdown("\n* **test MoCA (Montreal Cognitive Assessment)** to krótki test przesiewowy oceniający funkcje poznawcze, szeroko stosowany w diagnostyce łagodnych zaburzeń poznawczych i demencji.")
st.markdown("\n* **Craft Story 21 Recall** to test pamięci epizodycznej, w którym pacjent przypomina sobie 21-elementową historię po pewnym czasie opóźnienia, co pozwala ocenić zdolność przechowywania i przywoływania informacji werbalnych.")
st.markdown("\n* **Multilingual Naming Test (MINT)** to test nazywania obrazków przeznaczony dla osób wielojęzycznych, służący ocenie zdolności do przypominania sobie słów i ich nazywania. Jest przydatny w wykrywaniu zaburzeń językowych spowodowanych schorzeniami neurologicznymi.")

