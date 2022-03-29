import streamlit as st
import pickle as pk
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

class App:

    def __init__(self) -> None:
        pass
    def tratamento(self, dataframe_original):
        dataframe = dataframe_original.copy()

        # Extraindo colunas da feature categórica 'ed'
        dataframe['ed_1'] = dataframe['ed'].apply(lambda x:1 if x==1 else 0)
        dataframe.drop('ed', axis=1, inplace=True)

        # Guardando os dados de 'employ' em intervalos
        employ_min = 0
        employ_max = 31
        bins = np.linspace(employ_min, employ_max, 4) #4 pontos -> três intervalos
        group_names = [1, 2, 3]
        dataframe['employ'] = pd.cut(x=dataframe['employ'], bins=bins, labels=group_names, include_lowest=True)

        # Corrigindo desbalanceamento de 'employ'
        dataframe['employ_lvl_1'] = dataframe['employ'].apply(lambda x:1 if x==1 else 0)
        dataframe.drop('employ', axis=1, inplace=True)


        # Guardando os dados de 'address' em intervalos
        address_min = 0
        address_max = 34
        bins = np.linspace(address_min, address_max, 4) #4 pontos -> três intervalos
        group_names = [1, 2, 3]
        dataframe['address'] = pd.cut(x=dataframe['address'], bins=bins, labels=group_names, include_lowest=True)

        # Corrigindo desbalanceamento de 'address'
        dataframe['address_group_1'] = dataframe['address'].apply(lambda x:1 if x==1 else 0)
        dataframe.drop('address', axis=1, inplace=True)

        return dataframe


    def carrega_modelo(self):
        arq = open('../Modelos/modelo_classificacao_default.pk', 'rb')
        modelo = pk.load(arq)
        return modelo

    def proba_func(self, lista):
        modelo = self.carrega_modelo()
        dataframe = self.dataframe
        dataframe['debtinc'] = [lista[0]]
        dataframe['creddebt'] = [lista[1]]
        dataframe['othdebt'] = [lista[2]]
        
        return modelo.predict_proba(dataframe)[:,1][0]

    def atualiza_df(self, dataframe):
        self.dataframe = self.tratamento(dataframe)


if __name__ == '__main__':
    app = App()
    flag_otm = False
    modelo = app.carrega_modelo()

    st.markdown('# Aplicação')
    st.markdown('Aplicação para teste do modelo preditivo desenvolvido. Clique [aqui](https://github.com/ViniPilan/risk-prediction) para acesso ao projeto completo.')
    st.markdown('## Como usar a aplicação')
    st.markdown("Basta preencher os dados abaixo e clicar no botão 'calcular'. Após este processo, o resultado aparecerá logo abaixo juntamente com um campo para debug.")
    st.markdown('')

    dataframe = pd.DataFrame({'age':[int(st.slider('Idade', 0, 100))],
                              'ed':[int(st.slider('Nível de educação', 1, 5))],
                              'employ':[int(st.slider('Experiência de Trabalho', 0, 33))],
                              'address':[int(st.slider('Endereço do cliente', 0, 34))],
                              'income':[round(st.number_input('Salário anual (U$ - Dólar)'))//1000],
                              'debtinc':[float(st.number_input('Relação débito e salário do cliente'))],
                              'creddebt':[float(st.number_input('Relação crédito e débito do cliente'))],
                              'othdebt':[float(st.number_input('Outros débitos'))]})

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Calcular'):
            flag_otm = True  
            app.atualiza_df(dataframe)
            

    with col2:    
        botao_otm = st.button('Otimizar')
        if botao_otm and flag_otm:
            st.write('Calculando melhores parâmetros... Aguarde')
            space = [Real(0, 3, name='debtinc'),
                    Real(0, 1, name='creddebt'),
                    Real(0, 1, name='othdebt')]

            teste = gp_minimize(app.proba_func, space, random_state=7)

            st.write("Best parameters: \n- debtinc=%.2f \n- creddebt=%.2f \n- othdebt=%.2f" % (teste.x[0], teste.x[1], teste.x[2]))

        if botao_otm and flag_otm == False:
            st.write('Digite os dados e os classifique antes de fazer a otimização!')
        # FALTA ARRUMAR AS CLASSIFICAÇÕES
    
  