import pandas as pd
import streamlit as st
from io import BytesIO
from pycaret.classification import load_model, predict_model
import chardet

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    output = BytesIO()
    # Use openpyxl como engine ou xlsxwriter
    writer = pd.ExcelWriter(output, engine='openpyxl')  # Você pode trocar para 'xlsxwriter' se preferir
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()  # Alterado para close() em vez de save()
    processed_data = output.getvalue()
    return processed_data

def read_file(file):
    # Verifica a extensão do arquivo
    if file.name.endswith('.csv'):
        # Detecta a codificação e lê o CSV
        rawdata = file.read(1000)  
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        file.seek(0)  # Reverter o ponteiro do arquivo
        return pd.read_csv(file, encoding=encoding)
    
    elif file.name.endswith('.ftr'):
        # Ler arquivo .ftr (Feather) diretamente
        file.seek(0)  # Reverter o ponteiro do arquivo
        return pd.read_feather(file)
    
    else:
        st.error("Tipo de arquivo não suportado.")
        return None

def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title='PyCaret', page_icon = 'https://github.com/laysfelix/projetoPycaret/blob/e3ef7294d7b2898c08efac742491161ea31d2500/logo.jpg', layout="wide", initial_sidebar_state='expanded')

    st.write("""## Escorando o modelo gerado no pycaret """)
    st.markdown("---")
    
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    if data_file_1 is not None:
        df_credit = read_file(data_file_1)
        
        if df_credit is not None:
            df_credit = df_credit.sample(10000)

            model_saved = load_model('model_final')
            predict = predict_model(model_saved, data=df_credit)

            df_xlsx = to_excel(predict)
            st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')

if __name__ == '__main__':
    main()

