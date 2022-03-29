import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from skopt import gp_minimize
from skopt.space import Real

class App:
    def __init__(self):

        self.modelo = self.carrega_modelo()

        self.flag_otm = False

        st.markdown('# Aplicação de Predição de Risco')
        st.markdown('## Sobre os dados solicitados')
        st.markdown("""Os dados solicitados estão de acordo com o data set que foi utilizado durante o projeto. Portanto, para mais informações basta acessar 
                       a Análise Explorátória feita no projeto (clique [aqui](https://github.com/ViniPilan/risk-prediction/blob/master/Notebooks/AED.ipynb)).""")
        st.markdown('Aplicação para teste do modelo preditivo desenvolvido. Clique [aqui](https://github.com/ViniPilan/risk-prediction) para acesso ao projeto completo.')
        st.markdown('## Como usar a aplicação')
        st.markdown("Para cálculo de risco do cliente ser devedor, basta preencher os dados abaixo e clicar no botão 'Calcular risco'.")
        st.markdown("Para gerar os melhores parâmetros que diminuam o risco, clique em 'Otimizar parâmetros' após já ter inserido os dados e feito o cálculo do risco.")
        st.markdown('')

        col1, col2 = st.columns(2)

        with col1:
            slider_idade = st.slider('Idade', 18, 100)
            slider_nivel_edu = st.slider('Nível de educação', 0, 5)
            slider_exp_trab = st.slider('Experiência de Trabalho', 0, 33)
            slider_endereco = st.slider('Endereço do cliente', 0, 34)
            
        
        with col2:
            input_salario = st.number_input('Salário anual')
            input_deb_sal = st.number_input('Relação débito e salário anual do cliente')
            input_cred_deb = st.number_input('Relação crédito e débito do cliente')
            input_outros_deb = st.number_input('Outros débitos')


        col3, col4 = st.columns(2)

        with col3:
            self.dataframe = pd.DataFrame({'age':[int(slider_idade)],
                                'ed':[int(slider_nivel_edu)],
                                'employ':[int(slider_exp_trab)],
                                'address':[int(slider_endereco)],
                                'income':[round(input_salario)//1000],
                                'debtinc':[float(input_deb_sal)],
                                'creddebt':[float(input_cred_deb)],
                                'othdebt':[float(input_outros_deb)]})
            
            botao_calcular = st.button('Calcular risco')

            if botao_calcular:
                self.flag_otm = True
                st.dataframe(self.tratamento(self.dataframe))
                st.write(f'Probabilidade de devedor: {round(self.modelo.predict_proba(self.tratamento(self.dataframe))[:,1][0]*100, 2)}%')

        with col4:
            botao_otimizar = st.button('Otimizar parâmetros')

            if botao_otimizar:
                st.write('O resultado aparecerá logo abaixo assim que a otimização terminar (pode levar um tempo):')
                space = [Real(1, 3, name='debtinc'),
                            Real(0.2, 1, name='creddebt'),
                            Real(0.2, 1, name='othdebt')]
                teste = gp_minimize(self.proba_func, space, random_state=7)

                deb = teste.x[0]
                cred = teste.x[1]
                outros_deb = teste.x[2]

                if deb == 0:
                    deb = 'Mínimo possível'

                if cred == 0:
                    cred = 'Mínimo possível'

                if outros_deb == 0:
                    outros_deb = 'Mínimo possível'

                st.write("Melhores parâmetros: \n- debtinc={} \n- creddebt={} \n- othdebt={}".format(deb, cred, outros_deb))
                
               

        

    def carrega_modelo(self):
        arq = open('../Modelos/modelo_classificacao_default.pk', 'rb')
        modelo = pk.load(arq)
        return modelo

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

    def proba_func(self, lista):
        
        self.dataframe['debtinc'] = [lista[0]]
        self.dataframe['creddebt'] = [lista[1]]
        self.dataframe['othdebt'] = [lista[2]]
        
        return self.modelo.predict_proba(self.dataframe)[:,1][0]

if __name__ == '__main__':
    App()