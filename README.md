# Plante-ai
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Definição das classes na mesma ordem do treinamento
CLASSES = [
    'Acerola',
    'Ipê amarelo',
    'Ipê roxo',
    'Lajea rubra',
    'Magnólia',
    'Palmeira Imperial',
    'Sibipiruna'
]

# Dicionário com 2 imagens para cada planta. 
# Para usar suas próprias fotos, substitua a URL pelo caminho do arquivo local, 
# por exemplo: 'fotos/acerola_1.jpg'
GALERIA_IMAGENS = {
    'Acerola': [
        'https://placehold.co/600x400/e0f2f1/00695c?text=Acerola+-+Foto+1',
        'https://placehold.co/600x400/e0f2f1/00695c?text=Acerola+-+Foto+2'
    ],
    'Ipê amarelo': [
        'https://placehold.co/600x400/fffde7/f57f17?text=Ipe+Amarelo+-+Foto+1',
        'https://placehold.co/600x400/fffde7/f57f17?text=Ipe+Amarelo+-+Foto+2'
    ],
    'Ipê roxo': [
        'https://placehold.co/600x400/f3e5f5/6a1b9a?text=Ipe+Roxo+-+Foto+1',
        'https://placehold.co/600x400/f3e5f5/6a1b9a?text=Ipe+Roxo+-+Foto+2'
    ],
    'Lajea rubra': [
        'https://placehold.co/600x400/ffebee/c62828?text=Lajea+Rubra+-+Foto+1',
        'https://placehold.co/600x400/ffebee/c62828?text=Lajea+Rubra+-+Foto+2'
    ],
    'Magnólia': [
        'https://placehold.co/600x400/fce4ec/ad1457?text=Magnolia+-+Foto+1',
        'https://placehold.co/600x400/fce4ec/ad1457?text=Magnolia+-+Foto+2'
    ],
    'Palmeira Imperial': [
        'https://placehold.co/600x400/e8f5e9/2e7d32?text=Palmeira+Imperial+-+Foto+1',
        'https://placehold.co/600x400/e8f5e9/2e7d32?text=Palmeira+Imperial+-+Foto+2'
    ],
    'Sibipiruna': [
        'https://placehold.co/600x400/f1f8e9/558b2f?text=Sibipiruna+-+Foto+1',
        'https://placehold.co/600x400/f1f8e9/558b2f?text=Sibipiruna+-+Foto+2'
    ]
}

@st.cache_resource
def carregar_modelo():
    try:
        return tf.keras.models.load_model('modelo_plantas.h5')
    except Exception as e:
        return None

def preparar_imagem(imagem):
    if imagem.mode != "RGB":
        imagem = imagem.convert("RGB")
    imagem = imagem.resize((224, 224))
    imagem_array = tf.keras.preprocessing.image.img_to_array(imagem)
    imagem_array = np.expand_dims(imagem_array, axis=0)
    return imagem_array / 255.0

st.set_page_config(page_title="Identificador de Plantas", page_icon="🌿")
st.title('🌿 Classificador de Espécies Botânicas')
st.write('Faça o upload de uma foto da folha ou flor para identificar a espécie.')

modelo = carregar_modelo()
if modelo is None:
    st.warning("⚠️ O arquivo 'modelo_plantas.h5' não foi encontrado. A classificação não funcionará até que ele seja adicionado.")

arquivo_imagem = st.file_uploader("Escolha uma imagem (JPG, PNG)", type=["jpg", "jpeg", "png"])

if arquivo_imagem is not None and modelo is not None:
    imagem = Image.open(arquivo_imagem)
    st.image(imagem, caption='Imagem enviada', use_column_width=True)

    if st.button('Classificar Planta', type="primary"):
        with st.spinner('Analisando padrões biológicos...'):
            imagem_processada = preparar_imagem(imagem)
            predicoes = modelo.predict(imagem_processada)
            
            indice_classe = np.argmax(predicoes[0])
            confianca = np.max(predicoes[0]) * 100

            st.success(f"**Espécie identificada:** {CLASSES[indice_classe]}")
            st.info(f"Nível de confiança da IA: {confianca:.2f}%")

# --- GALERIA DE REFERÊNCIA ---
st.markdown("---")
st.subheader("📚 Galeria de Referência")
st.write("Compare a imagem enviada com os padrões de cada espécie:")

# Cria as abas de navegação para cada planta
abas = st.tabs(CLASSES)

# Preenche cada aba com 2 colunas contendo as imagens
for i, classe in enumerate(CLASSES):
    with abas[i]:
        col1, col2 = st.columns(2)
        with col1:
            st.image(GALERIA_IMAGENS[classe][0], caption=f'{classe} - Exemplo 1', use_column_width=True)
        with col2:
            st.image(GALERIA_IMAGENS[classe][1], caption=f'{classe} - Exemplo 2', use_column_width=True)
